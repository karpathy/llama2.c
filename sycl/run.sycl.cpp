/* Modifications Copyright (C) 2024 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 */

/* Inference for Llama-2 Transformer model in C++ + SYCL */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <fcntl.h>
#include <chrono>

// #include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#include <dpct/device.hpp>
using namespace dpct;

using dtype = sycl::half;
#define MAX_SEQ_LEN 2048
#define NEGATIVE_INFINITY -1 * std::numeric_limits<float>::infinity()
// ----------------------------------------------------------------------------
// Transformer model

typedef struct transformer_config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct transformer_weights {
    // token embedding table
    sycl::half *token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    sycl::half *rms_att_weight; // (layer, dim) rmsnorm weights
    sycl::half *rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    sycl::half *wq; // (layer, dim, n_heads * head_size)
    sycl::half *wk; // (layer, dim, n_kv_heads * head_size)
    sycl::half *wv; // (layer, dim, n_kv_heads * head_size)
    sycl::half *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    sycl::half *w1; // (layer, hidden_dim, dim)
    sycl::half *w2; // (layer, dim, hidden_dim)
    sycl::half *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    sycl::half *rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    sycl::half *freq_cis_real; // (seq_len, head_size/2)
    sycl::half *freq_cis_imag; // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    sycl::half *wcls;
} TransformerWeights;

typedef struct transformer_runstate {
    // current wave of activations
    sycl::half *x;   // activation at current time stamp (dim,)
    sycl::half *xb;  // same, but inside a residual branch (dim,)
    sycl::half *xb2; // an additional buffer just for convenience (dim,)
    sycl::half *hb;  // buffer for hidden dimension in the ffn (hidden_dim,)
    sycl::half *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    sycl::half *q;   // query (dim,)
    sycl::half *att; // buffer for scores/attention values (n_heads, seq_len)
    sycl::half *logits_gpu16; // output logits
    float* logits_gpu32; // logits in GPU memory converted to float
    float* logits; // logits copied CPU side
    // kv cache
    sycl::half *key_cache; // (layer, seq_len, dim)
    sycl::half *val_cache; // (layer, seq_len, dim)
} RunState;

typedef struct transformer_struct{
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
    int shared_weights;
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
    sycl::queue& q_ct1 = get_default_queue();

    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = p->dim / p->n_heads;

    // allocated on the GPU
    s->x            = sycl::malloc_device<sycl::half>(p->dim,                               q_ct1);
    s->xb           = sycl::malloc_device<sycl::half>(p->dim,                               q_ct1);
    s->xb2          = sycl::malloc_device<sycl::half>(p->dim,                               q_ct1);
    s->hb           = sycl::malloc_device<sycl::half>(p->hidden_dim,                        q_ct1);
    s->hb2          = sycl::malloc_device<sycl::half>(p->hidden_dim,                        q_ct1);
    s->q            = sycl::malloc_device<sycl::half>(p->n_heads * MAX_SEQ_LEN * head_size, q_ct1);
    s->att          = sycl::malloc_device<sycl::half>(p->n_heads  * p->seq_len,             q_ct1);
    s->logits_gpu16 = sycl::malloc_device<sycl::half>(p->vocab_size,                        q_ct1);
    s->logits_gpu32 = sycl::malloc_device<float>(p->vocab_size,                             q_ct1);
    s->key_cache    = sycl::malloc_device<sycl::half>(p->n_layers * p->seq_len * kv_dim,    q_ct1); // potentially huge allocs
    s->val_cache    = sycl::malloc_device<sycl::half>(p->n_layers * p->seq_len * kv_dim,    q_ct1);

    // allocated on the CPU
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->att || !s->key_cache || !s->val_cache
        || !s->logits_gpu16 || !s->logits_gpu32 || !s->logits) {
        fprintf(stderr, "malloc runstate failed!\n");
        exit(EXIT_FAILURE);
    }

    // q_ct1.memset(s->att, 0, p->n_heads * p->seq_len * sizeof(sycl::half));
    // q_ct1.memset(s->val_cache, 0, p->n_layers * p->seq_len * kv_dim * sizeof(sycl::half));
    // q_ct1.wait();
}

void free_run_state(RunState *s) {
    sycl::queue& q_ct1 = get_default_queue();

    sycl::free(s->x, q_ct1);
    sycl::free(s->xb, q_ct1);
    sycl::free(s->xb2, q_ct1);
    sycl::free(s->hb, q_ct1);
    sycl::free(s->hb2, q_ct1);
    sycl::free(s->q, q_ct1);
    sycl::free(s->att, q_ct1);
    sycl::free(s->logits_gpu16, q_ct1);
    sycl::free(s->logits_gpu32, q_ct1);
    sycl::free(s->key_cache, q_ct1);
    sycl::free(s->val_cache, q_ct1);
    free(s->logits);
}

void malloc_weights(TransformerWeights *w, Config *p, int shared_weights) {
    sycl::queue& q_ct1 = get_default_queue();

    int head_size = p->dim / p->n_heads;
    printf("p->vocab_size: %d\n", p->vocab_size);
    printf("p->dim: %d\n", p->dim);
    printf("p->hidden_dim: %d\n", p->hidden_dim);
    printf("p->n_layers: %d\n", p->n_layers);
    printf("p->n_heads: %d\n", p->n_heads);
    printf("p->n_kv_heads: %d\n", p->n_kv_heads);
    printf("p->seq_len: %d\n", p->seq_len);
    printf("head_size: %d\n", head_size);
    printf("shared_weights: %d\n", shared_weights);

    w->token_embedding_table = sycl::malloc_device<sycl::half>(p->vocab_size * p->dim,                              q_ct1);

    w->rms_att_weight        = sycl::malloc_device<sycl::half>(p->n_layers * p->dim,                                q_ct1);
    w->wq                    = sycl::malloc_device<sycl::half>(p->n_layers * p->dim * (p->n_heads * head_size),     q_ct1);
    w->wk                    = sycl::malloc_device<sycl::half>(p->n_layers * p->dim * (p->n_kv_heads * head_size),  q_ct1);
    w->wv                    = sycl::malloc_device<sycl::half>(p->n_layers * p->dim * (p->n_kv_heads * head_size),  q_ct1);
    w->wo                    = sycl::malloc_device<sycl::half>(p->n_layers * (p->n_heads * head_size) * p->dim,     q_ct1);

    w->rms_ffn_weight        = sycl::malloc_device<sycl::half>(p->n_layers * p->dim,                                q_ct1);
    w->w1                    = sycl::malloc_device<sycl::half>(p->n_layers * p->hidden_dim * p->dim,                q_ct1);
    w->w2                    = sycl::malloc_device<sycl::half>(p->n_layers * p->dim * p->hidden_dim,                q_ct1);
    w->w3                    = sycl::malloc_device<sycl::half>(p->n_layers * p->hidden_dim * p->dim,                q_ct1);

    w->rms_final_weight      = sycl::malloc_device<sycl::half>(p->dim,                                              q_ct1);

    w->freq_cis_real         = sycl::malloc_device<sycl::half>(p->seq_len * head_size / 2,                          q_ct1);
    w->freq_cis_imag         = sycl::malloc_device<sycl::half>(p->seq_len * head_size / 2,                          q_ct1);

    if (shared_weights) {
        w->wcls = w->token_embedding_table;
    } else {
        w->wcls              = sycl::malloc_device<sycl::half>(p->vocab_size * p->dim,                              q_ct1);
    }

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight
        || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 ||
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag || !w->wcls) {
        fprintf(stderr, "malloc weights failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_weights(TransformerWeights *w, int shared_weights) {
    sycl::queue& q_ct1 = get_default_queue();

    sycl::free(w->token_embedding_table, q_ct1);

    sycl::free(w->rms_att_weight, q_ct1);
    sycl::free(w->wq, q_ct1);
    sycl::free(w->wk, q_ct1);
    sycl::free(w->wv, q_ct1);
    sycl::free(w->wo, q_ct1);

    sycl::free(w->rms_ffn_weight, q_ct1);
    sycl::free(w->w1, q_ct1);
    sycl::free(w->w2, q_ct1);
    sycl::free(w->w3, q_ct1);

    sycl::free(w->rms_final_weight, q_ct1);

    sycl::free(w->freq_cis_real, q_ct1);
    sycl::free(w->freq_cis_imag, q_ct1);

    if (!shared_weights) {
        sycl::free(w->wcls, q_ct1);
    }
}

int load_weight(void* w, int elements, FILE* f, void* scratchCPU) {
    // read data into host memory
    int count = fread(scratchCPU, sizeof(dtype), elements, f);
    if (count != elements) return 1;
    // copy data to device memory
    get_default_queue().memcpy(w, scratchCPU, elements * sizeof(dtype)).wait();

    printf(".");
    fflush(stdout);
    return 0;
}

int load_checkpoint_weights(TransformerWeights *w, Config *p, FILE *f, int shared_weights) {
    sycl::queue& q_ct1 = get_default_queue();

    int head_size = p->dim / p->n_heads;
    size_t scratch_size = p->n_layers * std::max(p->dim, p->hidden_dim) * p->dim;
    scratch_size = std::max((size_t)p->vocab_size * p->dim, scratch_size);
    scratch_size *= sizeof(sycl::half);
    void* scratchCPU = malloc(scratch_size);

    printf("Loading weights\n");
     // populate each weight
    if (load_weight(w->token_embedding_table, p->vocab_size * p->dim,                               f, scratchCPU)) return 1;

    if (load_weight(w->rms_att_weight,        p->n_layers   * p->dim,                               f, scratchCPU)) return 1;
    if (load_weight(w->wq,                    p->n_layers   * p->dim * (p->n_heads    * head_size), f, scratchCPU)) return 1;
    if (load_weight(w->wk,                    p->n_layers   * p->dim * (p->n_kv_heads * head_size), f, scratchCPU)) return 1;
    if (load_weight(w->wv,                    p->n_layers   * p->dim * (p->n_kv_heads * head_size), f, scratchCPU)) return 1;
    if (load_weight(w->wo,                    p->n_layers   * (p->n_heads * head_size) * p->dim,    f, scratchCPU)) return 1;

    if (load_weight(w->rms_ffn_weight,        p->n_layers   * p->dim,                               f, scratchCPU)) return 1;
    if (load_weight(w->w1,                    p->n_layers   * p->dim * p->hidden_dim,               f, scratchCPU)) return 1;
    if (load_weight(w->w2,                    p->n_layers   * p->hidden_dim * p->dim,               f, scratchCPU)) return 1;
    if (load_weight(w->w3,                    p->n_layers   * p->dim * p->hidden_dim,               f, scratchCPU)) return 1;

    if (load_weight(w->rms_final_weight,      p->dim,                                               f, scratchCPU)) return 1;

    if (load_weight(w->freq_cis_real,         p->seq_len    * head_size / 2,                        f, scratchCPU)) return 1;
    if (load_weight(w->freq_cis_imag,         p->seq_len    * head_size / 2,                        f, scratchCPU)) return 1;

    if (!shared_weights) {
        if (load_weight(w->wcls,              p->vocab_size * p->dim,                               f, scratchCPU)) return 1;
    }

    printf("\ndone\n");
    free(scratchCPU);
    return 0;
}

void build_transformer(Transformer* t, char* checkpoint_path) {
    // open checkpoint
    FILE *file = fopen(checkpoint_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(EXIT_FAILURE); }
    // int magic, version;
    // if (fread(&magic,   sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    // if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(&t->config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    t->shared_weights = t->config.vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(t->config.vocab_size);
    // fseek(file, 256, SEEK_SET);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
    // allocate the Weights
    malloc_weights(&t->weights, &t->config, t->shared_weights);
    // read in the Config and the Weights from the checkpoint
    if (load_checkpoint_weights(&t->weights, &t->config, file, t->shared_weights)) { fprintf(stderr, "Couldn't load weights\n"); exit(EXIT_FAILURE); }
    fclose(file);
}

void free_transformer(Transformer* t) {
    // free the RunState buffers
    free_run_state(&t->state);
    // free the transformer weights
    free_weights(&t->weights, t->shared_weights);
}

// ----------------------------------------------------------------------------
// GPU kernels

SYCL_EXTERNAL void scalar_mul32_kernel(float *arr, float value, int size,
                                       const sycl::nd_item<1> &item_ct1) {
    int i = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);
    if (i < size)
        arr[i] = arr[i] * value;
}

SYCL_EXTERNAL void element_wise_add_kernel(sycl::half *dest, sycl::half *src,
                                           int size,
                                           const sycl::nd_item<1> &item_ct1) {
    int i = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);
    if (i < size)
        dest[i] = (sycl::half)((float)dest[i] + (float)src[i]);
}

SYCL_EXTERNAL void convert_fp32_to_fp16(sycl::half *out, float *in, int size,
                                        const sycl::nd_item<1> &item_ct1) {
    int index = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                item_ct1.get_local_id(0);
    if (index < size)
        out[index] = (sycl::half)in[index];
}

SYCL_EXTERNAL void convert_fp16_to_fp32(float *out, sycl::half *in, int size, const sycl::nd_item<1> &item_ct1) {
    int index = item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0);
    if (index < size)
        out[index] = (float)in[index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
SYCL_EXTERNAL void rmsnorm_kernel(dtype* o, const dtype* x, const dtype* weight, int size,
                    int elementsPerThread, const sycl::nd_item<1> &item_ct1,
                    float &shared_ss) {
    // float ss = 0.0f;
    // for (int index = item_ct1.get_local_id(0); index < size; index += 512) {
    //     float val = x[index];
    //     ss += (val * val);
    // }
    dtype ip[8];
    dtype wt[8];
    dtype vx[8];

    int j = item_ct1.get_local_id(0) * 8;

    *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&x[j]));
    *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));

    float ss = (float)ip[0] * (float)ip[0] + (float)ip[1] * (float)ip[1] + (float)ip[2] * (float)ip[2] + (float)ip[3] * (float)ip[3] + (float)ip[4] * (float)ip[4] + (float)ip[5] * (float)ip[5] + (float)ip[6] * (float)ip[6] + (float)ip[7] * (float)ip[7];

    ss = sycl::reduce_over_group(item_ct1.get_group(), ss, sycl::plus<>());

    if (item_ct1.get_local_id(0) == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sycl::sqrt(ss);
        shared_ss = ss;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    ss = shared_ss;

    // // normalize
    // for (int index = item_ct1.get_local_id(0); index < size; index += 512) {
    //     float val = ((float)x[index]) * ss * (float)weight[index];
    //     o[index] = (dtype)val;
    // }

    vx[0] = ip[0] * ss * wt[0];
    vx[1] = ip[1] * ss * wt[1];
    vx[2] = ip[2] * ss * wt[2];
    vx[3] = ip[3] * ss * wt[3];
    vx[4] = ip[4] * ss * wt[4];
    vx[5] = ip[5] * ss * wt[5];
    vx[6] = ip[6] * ss * wt[6];
    vx[7] = ip[7] * ss * wt[7];
    *((sycl::uint4 *)(&o[j])) = *((sycl::uint4 *)(&vx));
}

// Note that ~95% of total time is spent here, so optimizing this is important
// 1. One output generated per warp so that we can parallelize the dot product across the warp
// 2. We load 8 elements at a time for efficiency (assume dimensions to be multiple of 8)
SYCL_EXTERNAL void mat_vec_kernel(
    dtype* output,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weight,
    int n,
    int d,
    int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= d) return;

    input  += item.get_group(1) * input_stride;
    weight += item.get_group(1) * weight_stride + index * weight_row_stride;
    output += item.get_group(1) * output_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        if (j < n) {
            dtype wt[8];
            dtype ip[8];

            *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));
            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
            
            // for (int el = 0; el < 8; el++) {
            //     sum += float(wt[el]) * float(ip[el]);
            // }
            sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];
        }
    }

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

    // sum *= alpha;

    if (item.get_local_id(2) == 0) {
        output[index] = (dtype)sum;
    }
}

SYCL_EXTERNAL void mat_vec_f32_kernel(
    float* output,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weight,
    int n,
    int d,
    int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= d) return;

    input  += item.get_group(1) * input_stride;
    weight += item.get_group(1) * weight_stride + index * weight_row_stride;
    output += item.get_group(1) * output_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        if (j < n) {
            dtype wt[8];
            dtype ip[8];

            *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));
            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
            
            // for (int el = 0; el < 8; el++) {
            //     sum += float(wt[el]) * float(ip[el]);
            // }
            sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];
        }
    }

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

    // sum *= alpha;

    if (item.get_local_id(2) == 0) {
        output[index] = sum;
    }
}

SYCL_EXTERNAL void mat_vec_mad_kernel(
    dtype* output,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weight,
    int n,
    int d,
    int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= d) return;

    input  += item.get_group(1) * input_stride;
    weight += item.get_group(1) * weight_stride + index * weight_row_stride;
    output += item.get_group(1) * output_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        if (j < n) {
            dtype wt[8];
            dtype ip[8];

            *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));
            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
            
            // for (int el = 0; el < 8; el++) {
            //     sum += float(wt[el]) * float(ip[el]);
            // }
            sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];
        }
    }

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

    // sum *= alpha;

    if (item.get_local_id(2) == 0) {
        output[index] = (dtype)((float)output[index] + sum);
    }
}

// SYCL_EXTERNAL void mat_vec_qkv_kernel(
//     dtype* qout,
//     dtype* kout,
//     dtype* vout,
//     const dtype* __restrict__ input,
//     const dtype* __restrict__ weightq,
//     const dtype* __restrict__ weightk,
//     const dtype* __restrict__ weightv,
//     int n,
//     int d,
//     int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

//     int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
//     if (index >= d) return;

//     input   += item.get_group(1) *  input_stride;
//     weightq += item.get_group(1) * weight_stride + index * weight_row_stride;
//     weightk += item.get_group(1) * weight_stride + index * weight_row_stride;
//     weightv += item.get_group(1) * weight_stride + index * weight_row_stride;
//     qout    += item.get_group(1) * output_stride;
//     kout    += item.get_group(1) * output_stride;
//     vout    += item.get_group(1) * output_stride;

//     float sumq = 0;
//     float sumk = 0;
//     float sumv = 0;

//     for (int i = 0; i < numSerialLoads; i++) {
//         int j = (i * 32 + item.get_local_id(2)) * 8;

//         if (j < n) {
//             dtype ip[8];
//             dtype wq[8];
//             dtype wk[8];
//             dtype wv[8];

//             *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j]  ));
//             *((sycl::uint4 *)(&wq)) = *((sycl::uint4 *)(&weightq[j]));
//             *((sycl::uint4 *)(&wk)) = *((sycl::uint4 *)(&weightk[j]));
//             *((sycl::uint4 *)(&wv)) = *((sycl::uint4 *)(&weightv[j]));

//             // for (int el = 0; el < 8; el++) {
//             //     sumq += float(wq[el]) * float(ip[el]);
//             //     sumk += float(wk[el]) * float(ip[el]);
//             //     sumv += float(wv[el]) * float(ip[el]);
//             // }
//             sumq += wq[0] * ip[0] + wq[1] * ip[1] + wq[2] * ip[2] + wq[3] * ip[3] + wq[4] * ip[4] + wq[5] * ip[5] + wq[6] * ip[6] + wq[7] * ip[7];
//             sumk += wk[0] * ip[0] + wk[1] * ip[1] + wk[2] * ip[2] + wk[3] * ip[3] + wk[4] * ip[4] + wk[5] * ip[5] + wk[6] * ip[6] + wk[7] * ip[7];
//             sumv += wv[0] * ip[0] + wv[1] * ip[1] + wv[2] * ip[2] + wv[3] * ip[3] + wv[4] * ip[4] + wv[5] * ip[5] + wv[6] * ip[6] + wv[7] * ip[7];
//         }
//     }

//     sumq = sycl::reduce_over_group(item.get_sub_group(), sumq, sycl::plus<>());
//     sumk = sycl::reduce_over_group(item.get_sub_group(), sumk, sycl::plus<>());
//     sumv = sycl::reduce_over_group(item.get_sub_group(), sumv, sycl::plus<>());

//     // sum *= alpha;

//     if (item.get_local_id(2) == 0) {
//         qout[index] = (dtype)sumq;
//         kout[index] = (dtype)sumk;
//         vout[index] = (dtype)sumv;
//         // vout[index * MAX_SEQ_LEN] = (dtype)sumv; // change 2
//     }
// }

SYCL_EXTERNAL void mat_vec_qkv_kernel(
    dtype* qout,
    dtype* kout,
    dtype* vout,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weightq,
    const dtype* __restrict__ weightk,
    const dtype* __restrict__ weightv,
    int cols, int dh, int Nh, int numSerialLoads, const sycl::nd_item<3>& item) {

    int row_in_head = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    int index = item.get_group(1) * dh + row_in_head;
    if (index >= Nh * dh) return;

    weightq += index * cols;
    weightk += index * cols;
    weightv += index * cols;
    qout    += item.get_group(1) * MAX_SEQ_LEN * dh + row_in_head;
    kout    += item.get_group(1) * MAX_SEQ_LEN * dh + row_in_head;
    vout    += item.get_group(1) * MAX_SEQ_LEN * dh + row_in_head;

    float sumq = 0;
    float sumk = 0;
    float sumv = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        
        if (j < cols) {
            dtype ip[8];
            dtype wq[8];
            dtype wk[8];
            dtype wv[8];

            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j]  ));
            *((sycl::uint4 *)(&wq)) = *((sycl::uint4 *)(&weightq[j]));
            *((sycl::uint4 *)(&wk)) = *((sycl::uint4 *)(&weightk[j]));
            *((sycl::uint4 *)(&wv)) = *((sycl::uint4 *)(&weightv[j]));
            
            // for (int el = 0; el < 8; el++) {
            //     sumq += float(wq[el]) * float(ip[el]);
            //     sumk += float(wk[el]) * float(ip[el]);
            //     sumv += float(wv[el]) * float(ip[el]);
            // }
            sumq += wq[0] * ip[0] + wq[1] * ip[1] + wq[2] * ip[2] + wq[3] * ip[3] + wq[4] * ip[4] + wq[5] * ip[5] + wq[6] * ip[6] + wq[7] * ip[7];
            sumk += wk[0] * ip[0] + wk[1] * ip[1] + wk[2] * ip[2] + wk[3] * ip[3] + wk[4] * ip[4] + wk[5] * ip[5] + wk[6] * ip[6] + wk[7] * ip[7];
            sumv += wv[0] * ip[0] + wv[1] * ip[1] + wv[2] * ip[2] + wv[3] * ip[3] + wv[4] * ip[4] + wv[5] * ip[5] + wv[6] * ip[6] + wv[7] * ip[7];
        }
    }

    sumq = sycl::reduce_over_group(item.get_sub_group(), sumq, sycl::plus<>());
    sumk = sycl::reduce_over_group(item.get_sub_group(), sumk, sycl::plus<>());
    sumv = sycl::reduce_over_group(item.get_sub_group(), sumv, sycl::plus<>());

    if (item.get_local_id(2) == 0) {
        qout[0] = (dtype)sumq;
        kout[0] = (dtype)sumk;
        vout[0] = (dtype)sumv;
    }
}

SYCL_EXTERNAL void mat_vec_2X_kernel(
    dtype* out1,
    dtype* out2,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weighth1,
    const dtype* __restrict__ weighth2,
    int n_cols,
    int n_rows,
    int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= n_rows) return;

    input    += item.get_group(1) *  input_stride;
    weighth1 += item.get_group(1) * weight_stride + index * weight_row_stride;
    weighth2 += item.get_group(1) * weight_stride + index * weight_row_stride;
    out1     += item.get_group(1) * output_stride;
    out2     += item.get_group(1) * output_stride;

    float sumh1 = 0;
    float sumh2 = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        
        if (j < n_cols) {
            dtype ip[8];
            dtype w1[8];
            dtype w2[8];

            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j]  ));
            *((sycl::uint4 *)(&w1)) = *((sycl::uint4 *)(&weighth1[j]));
            *((sycl::uint4 *)(&w2)) = *((sycl::uint4 *)(&weighth2[j]));
            
            // for (int el = 0; el < 8; el++) {
            //     sumh1 += float(w1[el]) * float(ip[el]);
            //     sumh2 += float(w2[el]) * float(ip[el]);
            // }
            sumh1 += w1[0] * ip[0] + w1[1] * ip[1] + w1[2] * ip[2] + w1[3] * ip[3] + w1[4] * ip[4] + w1[5] * ip[5] + w1[6] * ip[6] + w1[7] * ip[7];
            sumh2 += w2[0] * ip[0] + w2[1] * ip[1] + w2[2] * ip[2] + w2[3] * ip[3] + w2[4] * ip[4] + w2[5] * ip[5] + w2[6] * ip[6] + w2[7] * ip[7];
        }
    }

    sumh1 = sycl::reduce_over_group(item.get_sub_group(), sumh1, sycl::plus<>());
    sumh2 = sycl::reduce_over_group(item.get_sub_group(), sumh2, sycl::plus<>());

    // sum *= alpha;

    if (item.get_local_id(2) == 0) {
        // out1[index] = (dtype)sumh1;
        // out2[index] = (dtype)sumh2;

        sumh1 *= 1.0 / (1.0 + sycl::native::exp(-sumh1));
        sumh1 *= sumh2;
        out1[index] = (dtype)sumh1;
    }
}

// // Simpler version of the above - to handle non multiple of 8 dimensions too (needed for MHA block)
// SYCL_EXTERNAL void mat_vec_kernel_simple(
//     dtype *output,
//     const dtype *__restrict__ input,
//     const dtype *__restrict__ weight,
//     int n,
//     int d,
//     int numSerialElements, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3> &item_ct1) {

//     int index = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
//     if (index >= d) return;

//     input  += item_ct1.get_group(1) * input_stride;
//     weight += item_ct1.get_group(1) * weight_stride + index * weight_row_stride;
//     output += item_ct1.get_group(1) * MAX_SEQ_LEN;

//     dtype sum(0);

//     // for (int i = 0; i < numSerialElements; i++) {
//     //     int j = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
//     //     if (j < n) {
//     //         sum += weight[j] * input[j];
//     //     }
//     // }

//     // int j = item_ct1.get_local_id(2) * 4;
//     // dtype ip[4];
//     // dtype wt[4];

//     // *((sycl::uint2 *)(&ip)) = *((sycl::uint2 *)(&input[j] ));
//     // *((sycl::uint2 *)(&wt)) = *((sycl::uint2 *)(&weight[j]));

//     // sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3];

//     int j = item_ct1.get_local_id(2) * 8;
//     dtype ip[8];
//     dtype wt[8];

//     *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
//     *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));

//     sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];

//     sum = sycl::reduce_over_group(item_ct1.get_sub_group(), sum, sycl::plus<>());
//     sum *= alpha;
//     if (item_ct1.get_local_id(2) == 0) {
//         output[index] = sum;
//     }
// }

// Simpler version of the above - to handle non multiple of 8 dimensions too (needed for MHA block)
SYCL_EXTERNAL void mat_vec_kernel_simple(
    dtype *output,
    const dtype *__restrict__ input,
    const dtype *__restrict__ weight,
    int dh,
    int seq_len,
    int numSerialElements, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3> &item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= seq_len) return;

    input  += item.get_group(1) * MAX_SEQ_LEN * dh;
    weight += item.get_group(1) * MAX_SEQ_LEN * dh + index * dh;
    output += item.get_group(1) * MAX_SEQ_LEN;

    dtype sum(0);

    // for (int i = 0; i < numSerialElements; i++) {
    //     int j = i * item.get_local_range(2) + item.get_local_id(2);
    //     if (j < dh) {
    //         sum += weight[j] * input[j];
    //     }
    // }

    // int j = item.get_local_id(2) * 4;
    // dtype ip[4];
    // dtype wt[4];

    // *((sycl::uint2 *)(&ip)) = *((sycl::uint2 *)(&input[j] ));
    // *((sycl::uint2 *)(&wt)) = *((sycl::uint2 *)(&weight[j]));
    
    // sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3];

    int j = item.get_local_id(2) * 8;
    dtype ip[8];
    dtype wt[8];

    *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
    *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));

    sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());
    sum *= alpha;
    if (item.get_local_id(2) == 0) {
        output[index] = sum;
    }
}

// // Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
// SYCL_EXTERNAL void vec_mat_kernel(dtype *output, const dtype *__restrict__ input,
//                const dtype *__restrict__ weight, int N, int K,
//                int elementsPerThread, int input_stride, int weight_stride,
//                int output_stride, int weight_row_stride,
//                const sycl::nd_item<3>& item,
//                sycl::local_accessor<dtype, 3> loaded_fragment) {

//     // locate beginning of my head
//     input  += item.get_group(1) * MAX_SEQ_LEN;      // h * max_seq_len
//     weight += item.get_group(1) * weight_stride;    // h * head_size for token 0
//     output += item.get_group(1) * output_stride;    // h * head_size

//     int start_n = item.get_group(2) * 32;           // item.get_group(2) -> 0, 1, 2, 3 -> which subgroup in this head
//     int i = start_n + item.get_local_id(1);         // item.get_local_id(1) -> 0, 1, 2, ..., 31

//     // 2x for double buffering
//     // +2 to avoid shared memory bank conflicts

//     // OOB check
//     if (i >= N)
//         return;

//     // load the first 32x32 fragment
//     int n = start_n + item.get_local_id(2);
//     int k = item.get_local_id(1);
//     int offset = k * weight_row_stride + n;
//     loaded_fragment[0][item.get_local_id(1)][item.get_local_id(2)] = ((n < N) && (k < K)) ? weight[offset] : (dtype)0;

//     float sum = 0;
//     // Loop over the matrix row and vector elements
//     for (int e = 0; e < elementsPerThread;)
//     {
//         item.barrier(sycl::access::fence_space::local_space); // wait for the load

//         int start_k = e * 32;
//         k = start_k + item.get_local_id(2);
//         int buf_i = e & 1;
//         sum += float(loaded_fragment[buf_i][item.get_local_id(2)][item.get_local_id(1)]) * (float)(input[k]);

//         // load for the next iteration
//         e++;
//         start_k = e * 32;
//         buf_i = e & 1;
//         n = start_n + item.get_local_id(2);
//         k = start_k + item.get_local_id(1);
//         int offset = k * weight_row_stride + n;
//         loaded_fragment[buf_i][item.get_local_id(1)][item.get_local_id(2)] = ((n < N) && (k < K)) ? weight[offset] : (dtype)0;
//     }

//     sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

//     if (item.get_local_id(2) == 0)
//         output[i] = (dtype)sum;
// }

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
SYCL_EXTERNAL void vec_mat_kernel(
    dtype *output, const dtype *__restrict__ input, const dtype *__restrict__ weight,
    int dh, int seq_len, int elementsPerThread, int input_stride, int weight_stride, int output_stride, int weight_row_stride,
    const sycl::nd_item<3>& item, sycl::local_accessor<dtype, 3> loaded_fragment) {

    // locate beginning of my head
    input  += item.get_group(1) * MAX_SEQ_LEN;      // h * max_seq_len
    weight += item.get_group(1) * MAX_SEQ_LEN * dh; // h * max_seq_len * head_size for token 0
    output += item.get_group(1) * dh;               // h * head_size

    int start_n = item.get_group(2) * 32;           // item.get_group(2) -> 0, 1, 2, 3 -> the subgroup in this head
    int i = start_n + item.get_local_id(1);         // item.get_local_id(1) -> 0, 1, 2, ..., 31

    // 2x for double buffering
    // +2 to avoid shared memory bank conflicts

    // OOB check
    if (i >= dh)
        return;

    // load the first 32x32 fragment
    int n = start_n + item.get_local_id(2);
    int k = item.get_local_id(1);
    int offset = k * dh + n;
    loaded_fragment[0][item.get_local_id(1)][item.get_local_id(2)] = ((n < dh) && (k < seq_len)) ? weight[offset] : (dtype)0;

    float sum = 0;
    // Loop over the matrix row and vector elements
    for (int e = 0; e < elementsPerThread;)
    {
        item.barrier(sycl::access::fence_space::local_space); // wait for the load

        int start_k = e * 32;
        k = start_k + item.get_local_id(2);
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][item.get_local_id(2)][item.get_local_id(1)]) * (float)(input[k]);

        // load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + item.get_local_id(2);
        k = start_k + item.get_local_id(1);
        int offset = k * dh + n;
        loaded_fragment[buf_i][item.get_local_id(1)][item.get_local_id(2)] = ((n < dh) && (k < seq_len)) ? weight[offset] : (dtype)0;
    }

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

    if (item.get_local_id(2) == 0)
        output[i] = (dtype)sum;
}

// // Each block processes a single head
// SYCL_EXTERNAL void RoPERotation_kernel(dtype* sq, dtype* sk, int pos, int num_heads, int num_kv_heads, int head_size, const sycl::nd_item<1> &item) {
//     int h = item.get_group(0);

//     dtype *q = sq + h * head_size;
//     dtype *k = sk + h * head_size;

//     int i = item.get_local_id(0) * 2;

//     int head_dim = i % head_size;
//     float freq = 1.0f / sycl::pow(10000.0f, head_dim / (float)head_size);
//     float val = pos * freq;
//     float fcr = sycl::cos(val);
//     float fci = sycl::sin(val);

//     // rotate q
//     float q0 = q[i];
//     float q1 = q[i + 1];
//     q[i]     = q0 * fcr - q1 * fci;
//     q[i + 1] = q0 * fci + q1 * fcr;

//     // rotate k
//     if (h < num_kv_heads) {
//         float k0 = k[i];
//         float k1 = k[i + 1];
//         k[i]     = k0 * fcr - k1 * fci;
//         k[i + 1] = k0 * fci + k1 * fcr;
//     }
// }

// Each block processes a single head
SYCL_EXTERNAL void RoPERotation_kernel(
    dtype* sq, dtype* sk,
    int pos, int num_heads, int num_kv_heads, int head_size, const sycl::nd_item<1> &item) {

    int h = item.get_group(0);

    dtype *q = sq + (h * MAX_SEQ_LEN) * head_size;
    dtype *k = sk + (h * MAX_SEQ_LEN) * head_size;

    int i = item.get_local_id(0) * 2;

    int head_dim = i % head_size;
    float freq = 1.0f / sycl::pow(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = sycl::cos(val);
    float fci = sycl::sin(val);

    // rotate q
    float q0 = q[i];
    float q1 = q[i + 1];
    q[i]     = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    // rotate k
    if (h < num_kv_heads) {
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i]     = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

SYCL_EXTERNAL void softmax_kernel(dtype *__restrict__ arr, int num_heads,
                                  int size, const sycl::nd_item<1>& item,
                                  float *att, float &shared_val) {

    int h = item.get_group(0);
    int tid = item.get_local_id(0);
    int step = item.get_local_range(0);

    dtype *__restrict__ arr_base = arr + h * MAX_SEQ_LEN;

    // load input to shared memory
    for (int t = tid; t < size; t += step) {
        att[t] = (float)arr_base[t];
    }
    item.barrier(sycl::access::fence_space::local_space);

    // find max value (for numerical stability)
    float max = tid < size ? att[tid] : NEGATIVE_INFINITY;
    for (int i = tid + step; i < size; i += step) {
        auto temp = att[i];
        if (temp > max) max = temp;
    }
    max = sycl::reduce_over_group(item.get_group(), max, sycl::maximum<>());

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        auto temp = sycl::native::exp(att[i] - max);
        att[i] = temp;
        sum += temp;
    }
    sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<>());

    // normalize and write the result
    float inv_sum = 1.0f / sum;
    for (int t = tid; t < size; t += step) {
        arr_base[t] = (dtype)(att[t] * inv_sum);
    }
}

SYCL_EXTERNAL void softmax32_kernel(float *__restrict__ x, int size,
                                    const sycl::nd_item<1> &item,
                                    float &shared_val) {

    int tid = item.get_local_id(0);
    int step = item.get_local_range(0);

    // find max value (for numerical stability)
    float max = tid < size ? x[tid] : NEGATIVE_INFINITY;
    for (int i = tid + step; i < size; i += step) {
        auto temp = x[i];
        if (temp > max) max = temp;
    }
    max = sycl::reduce_over_group(item.get_group(), max, sycl::maximum<>());

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        auto temp = sycl::native::exp(x[i] - max);
        x[i] = temp;
        sum += temp;
    }
    sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<>());

    // normalize
    for (int i = tid; i < size; i += step)
        x[i] /= sum;
}

SYCL_EXTERNAL void argmax32_kernel(float* __restrict__ x, int size, int* result, const sycl::nd_item<1>& item) {

    int tid = item.get_local_id(0);
    int step = item.get_local_range(0);

    // find local max value and its position
    float max_val = tid < size ? x[tid] : 0;
    int   max_pos = tid < size ? tid : 0;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_pos = i;
        }
    }
    // find the global max value
    float global_max_val = sycl::reduce_over_group(item.get_group(), max_val, sycl::maximum<>());

    // get its position
    if (max_val == global_max_val) {
        *result = max_pos;
    }
}

SYCL_EXTERNAL void silu_element_wise_mul_kernel(dtype *dest, dtype *src, int size, const sycl::nd_item<1> &item) {
    int i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i < size) {
        float val = (float)dest[i];
        val *= 1.0f / (1.0f + sycl::native::exp(-val));
        val *= (float)src[i];
        dest[i] = (dtype)val;
    }
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

void accum(dtype *a, dtype *b, int size) {
    int blocks = divUp(size, 1024);
    {
        get_default_queue().parallel_for<class kernel_accum>(
            sycl::nd_range(sycl::range(blocks) * sycl::range(1024), sycl::range(1024)),
            [=](sycl::nd_item<1> item_ct1) {
                element_wise_add_kernel(a, b, size, item_ct1);
            });
    }
}

void rmsnorm(dtype *o, const dtype *x, const dtype *weight, int size) {
    int elementsPerThread = divUp(size, 512);
    {
        get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 0> shared_ss_acc_ct1(cgh);

            cgh.parallel_for<class kernel_rmsnorm>(
                sycl::nd_range(sycl::range(512), sycl::range(512)),
                [=](sycl::nd_item<1> item_ct1) [[intel::kernel_args_restrict]] {
                    rmsnorm_kernel(o, x, weight, size, elementsPerThread, item_ct1, shared_ss_acc_ct1);
                });
        });
    }
}

void matmul(
    dtype* xout, dtype* x,
    dtype* w,
    int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range block_dim(1, 4, 32);
    sycl::range grid_dim(1, batch, divUp(d, 4));
    if (w_row_stride == -1) w_row_stride = n;
    
    get_default_queue().parallel_for<class kernel_matmul>(
        sycl::nd_range(grid_dim * block_dim, block_dim),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_kernel(xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, item_ct1);
        }
    );
}

// void onemkl_gemm_rm(sycl::queue& queue,
//                    oneapi::mkl::transpose transa,
//                    oneapi::mkl::transpose transb,
//                    int m,
//                    int n,
//                    int k,
//                    const dtype  alpha,
//                    const dtype  beta,
//                    const dtype* A,
//                    const dtype* B,
//                          dtype* C)
// {
//     try {
//         int lda = (transa == oneapi::mkl::transpose::nontrans) ? k : m;
//         int ldb = (transb == oneapi::mkl::transpose::nontrans) ? n : k;
//         int ldc = n;

//         oneapi::mkl::blas::row_major::gemm(queue, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

//     } catch (sycl::exception const& exc) {
//         std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
//                   << std::endl;
//         std::exit(1);
//     }
// }

// void matmul_x(
//     dtype* xout, dtype* x,
//     dtype* w,
//     int K, int N) {

//     dtype alpha(1.0f);
//     dtype beta (0.0f);

//     onemkl_gemm_rm( get_default_queue(),
//                     oneapi::mkl::transpose::nontrans,
//                     oneapi::mkl::transpose::trans,
//                     1,  // N
//                     N,  // 1
//                     K,  // K
//                     alpha,
//                     beta,
//                     x,  // w
//                     w,  // x (faster)
//                     xout);
// }

void matmul_f32(
    float* xout, dtype* x,
    dtype* w,
    int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range block_dim(1, 4, 32);
    sycl::range grid_dim(1, batch, divUp(d, 4));
    if (w_row_stride == -1) w_row_stride = n;
    
    get_default_queue().parallel_for<class kernel_matmul_f32>(
        sycl::nd_range(grid_dim * block_dim, block_dim),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_f32_kernel(xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, item_ct1);
        }
    );
}

void matmul_mad(
    dtype* xout, dtype* x,
    dtype* w,
    int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range block_dim(1, 4, 32);
    sycl::range grid_dim(1, batch, divUp(d, 4));
    if (w_row_stride == -1) w_row_stride = n;
    
    get_default_queue().parallel_for<class kernel_matmul_mad>(
        sycl::nd_range(grid_dim * block_dim, block_dim),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_mad_kernel(xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, item_ct1);
        }
    );
}

// void matmul_qkv(
//     dtype* qout, dtype* kout, dtype* vout, dtype* x,
//     dtype* wq, dtype* wk, dtype* wv,
//     int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
//     int serialElements = divUp(n, 32);
//     int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
//     sycl::range block_dim(1, 4, 32);
//     sycl::range grid_dim(1, batch, divUp(d, 4));
//     if (w_row_stride == -1) w_row_stride = n;

//     get_default_queue().parallel_for<class kernel_matmul_qkv>(
//         sycl::nd_range(grid_dim * block_dim, block_dim),
//         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
//             mat_vec_qkv_kernel(qout, kout, vout, x, wq, wk, wv, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, item_ct1);
//         }
//     );
// }

void matmul_qkv(
    dtype* qout, dtype* kout, dtype* vout, const dtype* x,
    const dtype* wq, const dtype* wk, const dtype* wv,
    int cols, int dh, int Nh) {

    int serialElements = divUp(cols, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range local(1, 4, 32);
    sycl::range global(1, Nh, divUp(dh, 4));

    get_default_queue().parallel_for<class kernel_matmul_qkv>(
        sycl::nd_range(global * local, local),
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_qkv_kernel(qout, kout, vout, x, wq, wk, wv, cols, dh, Nh, serialLoads, item);
        }
    );
}

void matmul_2X(
    dtype* out1, dtype* out2, dtype* input, dtype* w1, dtype* w2, int n_cols, int n_rows,
    int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    int serialElements = divUp(n_cols, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range block_dim(1, 4, 32);
    sycl::range grid_dim(1, batch, divUp(n_rows, 4));
    if (w_row_stride == -1) w_row_stride = n_cols;

    get_default_queue().parallel_for<class kernel_matmul_w1w3>(
        sycl::nd_range(grid_dim * block_dim, block_dim),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_2X_kernel(out1, out2, input, w1, w2, n_cols, n_rows, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, item_ct1);
        }
    );
}

void RoPERotation(dtype *q, dtype *k, int pos, int num_heads, int num_kv_heads, int head_size) {
    get_default_queue().parallel_for<class kernel_rope>(
        sycl::nd_range(sycl::range(num_heads) * sycl::range(head_size / 2), sycl::range(head_size / 2)),
        [=](sycl::nd_item<1> item_ct1) {
            RoPERotation_kernel(q, k, pos, num_heads, num_kv_heads, head_size, item_ct1);
        });
}

void debug_print(dtype* device_data, int num_points = 128) {
    dtype* host_data = (dtype*)malloc(num_points * sizeof(dtype));
    get_default_queue().memcpy(host_data, device_data, num_points * sizeof(dtype)).wait();
    for (int i = 0; i < num_points; i++) {
        printf("%f ", (float)host_data[i]);
    }
    printf("\n");
    free(host_data);
}

// void writeData(dtype* device_data, std::string filename, const int rows, const int cols)
// {
//     std::ofstream out;
//     out.open(filename);

//     dtype* host_data = (dtype*)malloc(rows * cols * sizeof(dtype));
//     get_default_queue().memcpy(host_data, device_data, rows * cols * sizeof(dtype)).wait();

//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             out << (float)host_data[i * cols + j] << ",";
//         }
//         out << std::endl;
//     }

//     free(host_data);
//     out.close();
// }

void MultiHeadAttention(dtype *output,
                        dtype *q, dtype *key_cache, dtype *val_cache,
                        dtype *att,
                        int num_heads, int head_size, int seq_len) {

    sycl::queue& q_ct1 = get_default_queue();

    int dim = head_size * num_heads;

    // 1. Get attention scores
    constexpr int blockDim_x = 16;
    constexpr int blockDim_y = 8;
    constexpr int blockDim_z = 1;
              int gridDim_x  = divUp(seq_len, blockDim_y);
    constexpr int gridDim_y  = 32; // num_heads
    constexpr int gridDim_z  = 1;

    sycl::range block_dim1(blockDim_z, blockDim_y, blockDim_x);
    sycl::range grid_dim1(gridDim_z, gridDim_y, gridDim_x);
    int serialElements1 = divUp(head_size, blockDim_x);
    {
        q_ct1.submit([&](sycl::handler &cgh) {
            float sqrt_head_size = 1.0 / sqrt(head_size);

            cgh.parallel_for<class kernel_mat_vec_simple>(
                sycl::nd_range(grid_dim1 * block_dim1, block_dim1),
                [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(blockDim_x)]] {
                    mat_vec_kernel_simple(
                        att, q, key_cache,
                        head_size, seq_len, serialElements1, head_size, head_size, seq_len, dim, sqrt_head_size,
                        item_ct1);
                });
        });
    }

    // 2. Run softmax kernel
    {
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> att_acc_ct1(sycl::range(MAX_SEQ_LEN), cgh);
            sycl::local_accessor<float, 0> shared_val_acc_ct1(cgh);

            cgh.parallel_for<class kernel_softmax>(
                sycl::nd_range(sycl::range(num_heads) * sycl::range(1024), sycl::range(1024)),
                [=](sycl::nd_item<1> item_ct1) {
                    softmax_kernel(att, num_heads, seq_len, item_ct1, att_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get() , shared_val_acc_ct1);
                });
        });
    }

    // 3. weighted sum of the values to get the final result
    // constexpr int blockDim_x = 32;
    // constexpr int blockDim_y = 32;
    // constexpr int blockDim_z = 1;
    //           int gridDim_x  = divUp(head_size, blockDim_y);
    // constexpr int gridDim_y  = 32; // num_heads
    // constexpr int gridDim_z  = 1;
    int serialElements2 = divUp(seq_len, 32);
    sycl::range block_dim(1, 32, 32);
    sycl::range grid_dim2(1, num_heads, divUp(head_size, 32));
    {
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<dtype, 3> loaded_fragment_acc_ct1(sycl::range(2, 32, 32 + 2), cgh);

            cgh.parallel_for<class kernel_vec_mat>(
                sycl::nd_range(grid_dim2 * block_dim, block_dim),
                [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                    vec_mat_kernel(
                        output, att, val_cache,
                        head_size, seq_len, serialElements2, seq_len, head_size, head_size, dim,
                        item_ct1, loaded_fragment_acc_ct1);
                });
        });
    }

    // // change 3
    // {
    //     constexpr int blockDim_x = 32;
    //     constexpr int blockDim_y = 32;
    //     constexpr int blockDim_z = 1;
    //               int gridDim_x  = 32; // num_heads
    //     constexpr int gridDim_y  = 1;
    //     constexpr int gridDim_z  = 1;
    //     int numSerialLoads = divUp(seq_len, blockDim_x);
    //     sycl::range block_dim2(blockDim_z, blockDim_y, blockDim_x);
    //     sycl::range grid_dim2(gridDim_z, gridDim_y, gridDim_x);
    
    //     q_ct1.submit([&](sycl::handler &cgh) {
    //         cgh.parallel_for<class kernel_vec_mat>(
    //             sycl::nd_range(grid_dim2 * block_dim2, block_dim2),
    //             [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(blockDim_x)]] {
    //                 vec_mat_kernelx(output, att, val_cache, numSerialLoads, seq_len, item);
    //             });
    //     });
    // }
}

void siluElementwiseMul(dtype *hb, dtype *hb2, int size) {
    get_default_queue().parallel_for<class kernel_silu_eltwise_mul>(
        sycl::nd_range(sycl::range(divUp(size, 1024)) * sycl::range(1024), sycl::range(1024)),
        [=](sycl::nd_item<1> item_ct1) {
            silu_element_wise_mul_kernel(hb, hb2, size, item_ct1);
        }
    );
}

void forward(Transformer* transformer, int token, int pos) {

    sycl::queue& q_ct1 = get_default_queue();

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    dtype *x = s->x;

    // copy the token embedding into x
    dtype *content_row = &(w->token_embedding_table[token * dim]);
    q_ct1.memcpy(x, content_row, dim * sizeof(dtype));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // we directly store (key, value) at this time step (pos) to our kv cache
        int loff = l * p->n_kv_heads * p->seq_len * head_size; // kv cache layer offset for convenience
        dtype *qrow = s->q + pos * head_size;
        dtype *krow = s->key_cache + loff + pos * head_size;
        dtype *vrow = s->val_cache + loff + pos * head_size;

        // qkv matmuls for this position
        // matmul(qrow, s->xb, w->wq + l*dim*   dim, dim,    dim);
        // matmul(krow, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        // matmul(vrow, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        matmul_qkv(qrow, krow, vrow, s->xb, w->wq + l*dim*dim, w->wk + l*dim*kv_dim, w->wv + l*dim*kv_dim, dim, head_size, p->n_heads);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(qrow, krow, pos, p->n_heads, p->n_kv_heads, head_size);

        // apply MHA using the query and the key-value cache
        MultiHeadAttention(s->xb, qrow, s->key_cache + loff, s->val_cache + loff, s->att, p->n_heads, head_size, pos+1);

        // final matmul to get the output of the attention
        // matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        // accum(x, s->xb2, dim);
        matmul_mad(x, s->xb, w->wo + l*dim*dim, dim, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        // matmul(s->hb,  s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        // matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        matmul_2X(s->hb, s->hb2, s->xb, w->w1 + l*dim*hidden_dim, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // apply F.silu activation on hb and multiply it with hb2
        // siluElementwiseMul(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
        // matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        // accum(x, s->xb, dim);
        matmul_mad(x, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
    }

    // final rmsnorm
    rmsnorm(s->xb, x, w->rms_final_weight, dim);

    // classifier into logits
    // matmul(s->logits_gpu16, s->xb, w->wcls, p->dim, p->vocab_size);
    matmul_f32(s->logits_gpu32, s->xb, w->wcls, p->dim, p->vocab_size);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct transformer_tokenindex {
    char *str;
    int id;
} TokenIndex;

typedef struct transformer_tokenizer {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (len <= 0 || len > 200) { exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex*) malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*) malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char*)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct transformer_probindex{
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct transformer_sampler{
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
    sycl::queue& q_ct1 = get_default_queue();

    // return the index that has the highest probability
    int max_pos;
    int *pmax_pos;

    // allocate memory on the device
    pmax_pos = sycl::malloc_device<int>(1, q_ct1);

    // call the kernel
    q_ct1.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class kernel_argmax32>(
            sycl::nd_range(sycl::range(1024), sycl::range(1024)),
            [=](sycl::nd_item<1> item) {
                argmax32_kernel(probabilities, n, pmax_pos, item);
        });
    });

    // copy the result back to host
    q_ct1.memcpy(&max_pos, pmax_pos, sizeof(int)).wait();

    // free the allocated memory
    sycl::free(pmax_pos, q_ct1);

    return max_pos;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex*) malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, RunState* state) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(state->logits_gpu32, sampler->vocab_size);
    } else {
        // // apply the temperature to the logits
        // float inv_temperature = 1.0f / sampler->temperature;
        // scalar_mul32_kernel <<< divUp(sampler->vocab_size, 256), 256 >>> (state->logits_gpu32, inv_temperature, sampler->vocab_size);
        // // apply softmax to the logits to get the probabilities for next token
        // softmax32_kernel <<< 1, 1024 >>> (state->logits_gpu32, sampler->vocab_size);
        // // copy the logits from GPU to the CPU
        // cudaMemcpy(state->logits, state->logits_gpu32, sampler->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(state->logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(state->logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

struct struct_color
{
    const std::string GREEN = "\033[92m";
    const std::string BLUE  = "\033[94m";
    const std::string END   = "\033[0m";
} colors;

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = (char*)"";
    if (prompt == NULL) { prompt = empty_prompt; }
    if (strlen(prompt) > 1024) { exit(EXIT_FAILURE); }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    std::chrono::steady_clock::time_point time_start0;
    std::chrono::steady_clock::time_point time_start1;
    std::chrono::steady_clock::time_point time_start2;
    std::chrono::steady_clock::time_point time_start3;
    std::chrono::steady_clock::time_point time_stop;

    // start the main loop
    // long start0 = 0;  // used to time our code, only initialized after first iteration
    // long start1 = 0;  // used to time our code, only initialized after first iteration
    // long start2 = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence

    // start0 = time_in_ms();
    time_start0 = std::chrono::steady_clock::now();

    while (pos < steps) {

        // forward the transformer to get logits for the next token
        forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, &transformer->state);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        // fflush(stdout);
        if (pos < num_prompt_tokens) {
            std::cout << colors.BLUE << piece << colors.END << std::flush;
        } else {
            std::cout << colors.GREEN << piece << colors.END << std::flush;
        }
        token = next;

        // init the timer here because the first iteration can be slower
        // if (start2 == 0) { start2 = time_in_ms(); }
        if (pos == num_prompt_tokens) {
            // start1 = time_in_ms();
            time_start1 = std::chrono::steady_clock::now();
        }
        if (pos == num_prompt_tokens + 1) {
            // setenv("PTI_ENABLE_COLLECTION", "1", 1);
            // start2 = time_in_ms();
            time_start2 = std::chrono::steady_clock::now();
        }
        if (pos == num_prompt_tokens + 2) {
            time_start3 = std::chrono::steady_clock::now();
        }
    }
    printf("\n");
    // unsetenv("PTI_ENABLE_COLLECTION");
    // report achieved tok/s (pos - num_prompt_tokens because the timer starts after first iteration)
    if (pos > num_prompt_tokens) {
        // long end = time_in_ms();
        time_stop = std::chrono::steady_clock::now();
        printf("input tokens: %d\n", num_prompt_tokens);
        printf("new tokens: %d\n", pos - num_prompt_tokens);

        // printf("achieved 1st token latency: %f ms\n", (double)(start2 - start1));
        // printf("achieved 1st token latency (incl input): %f ms\n", (double)(start2 - start0) / (num_prompt_tokens + 1));

        // printf("achieved tok/s: %f\n", (pos - num_prompt_tokens) / (double)(end - start2) * 1000);
        // printf("achieved next token latency: %f ms\n", (double)(end - start2) / (pos - num_prompt_tokens - 1));


        printf("achieved 1st token latency: %f ms\n", std::chrono::duration<double, std::milli>(time_start2 - time_start1).count());
        // printf("achieved 1st token latency (incl input): %f ms\n", std::chrono::duration<double, std::milli>(time_start2 - time_start0).count() / (num_prompt_tokens + 1));

        printf("achieved 2nd token latency: %f ms\n", std::chrono::duration<double, std::milli>(time_start3 - time_start2).count());
        // printf("achieved next token latency: %f ms\n", std::chrono::duration<double, std::milli>(time_stop - time_start2).count() / (pos - num_prompt_tokens - 1));
    }

    free(prompt_tokens);
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
try {
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = (char*)"tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = (char*)"generate";  // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned long long)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    // warmup
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    //profile
    // setenv("PTI_ENABLE_COLLECTION", "1", 1);
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    // unsetenv("PTI_ENABLE_COLLECTION");

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
} catch(...) {
    printf("Ran into an exception\n");
    return 1;
}
}
