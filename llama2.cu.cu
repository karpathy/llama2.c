/*
Inference for Llama-2 Transformer model in pure Cuda.
*/

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    half* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    half* rms_att_weight; // (layer, dim) rmsnorm weights
    half* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    half* wq; // (layer, dim, dim)
    half* wk; // (layer, dim, dim)
    half* wv; // (layer, dim, dim)
    half* wo; // (layer, dim, dim)
    // weights for ffn
    half* w1; // (layer, hidden_dim, dim)
    half* w2; // (layer, dim, hidden_dim)
    half* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    half* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    half* freq_cis_real; // (seq_len, dim/2)
    half* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    half* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    half* x; // activation at current time stamp (dim,)
    half* xb; // same, but inside a residual branch (dim,)
    half* xb2; // an additional buffer just for convenience (dim,)
    half* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* q; // query (dim,)
    half* k; // key (dim,)
    half* v; // value (dim,)
    half* logits_gpu; // output logits
    // kv cache
    half* key_cache;   // (layer, seq_len, dim)
    half* value_cache; // (layer, seq_len, dim)

    float* logits;     // logits copied CPU side (and converted to float)
    float* logits_temp;
} RunState;

void malloc_run_state(RunState* s, Config* p) {
    cudaMalloc((void**)&s->x, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb2, p->dim * sizeof(half));
    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->q, p->dim * sizeof(half));
    cudaMalloc((void**)&s->k, p->dim * sizeof(half));
    cudaMalloc((void**)&s->v, p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(half));
    cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * p->dim * sizeof(half));    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits_temp, p->vocab_size * sizeof(float));
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->k || !s->v || !s->logits || !s->key_cache
        || !s->value_cache || !s->logits_gpu) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->k);
    cudaFree(s->v);
    cudaFree(s->logits_gpu);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
    cudaFree(s->logits_temp);
    free(s->logits);
}

void malloc_weights(TransformerWeights* w, Config* p, int shared_weights) {
    cudaMalloc((void**)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(half));
    cudaMalloc((void**)&w->rms_att_weight, p->n_layers * p->dim * sizeof(half));
    cudaMalloc((void**)&w->rms_ffn_weight, p->n_layers * p->dim * sizeof(half));
    cudaMalloc((void**)&w->wq, p->n_layers * p->dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->wk, p->n_layers * p->dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->wv, p->n_layers * p->dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->wo, p->n_layers * p->dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->w1, p->n_layers * p->hidden_dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->w2, p->n_layers * p->dim * p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&w->w3, p->n_layers * p->hidden_dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->rms_final_weight, p->dim * sizeof(half));
    int head_size = p->dim / p->n_heads;
    cudaMalloc((void**)&w->freq_cis_real, p->seq_len * head_size / 2 * sizeof(half));
    cudaMalloc((void**)&w->freq_cis_imag, p->seq_len * head_size / 2 * sizeof(half));

    if (shared_weights)
        w->wcls = w->token_embedding_table;
    else
        cudaMalloc((void**)&w->wcls, p->vocab_size * p->dim * sizeof(half));

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight
        || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 ||
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag || !w->wcls) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_weights(TransformerWeights* w, int shared_weights) {
    cudaFree(w->token_embedding_table);
    cudaFree(w->rms_att_weight);
    cudaFree(w->rms_ffn_weight);
    cudaFree(w->wq);
    cudaFree(w->wk);
    cudaFree(w->wv);
    cudaFree(w->wo);
    cudaFree(w->w1);
    cudaFree(w->w2);
    cudaFree(w->w3);
    cudaFree(w->rms_final_weight);
    cudaFree(w->freq_cis_real);
    cudaFree(w->freq_cis_imag);
    if (!shared_weights)
        cudaFree(w->wcls);
}

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

__global__ void ConvertFP32toFP16(half *out, float *in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (half)in[index];
}

__global__ void ConvertFP16toFP32(float* out, half* in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (float)in[index];
}

int uploadWeight(void *w, int elements, FILE* f, void *scratchCpu, void *scratchGpu) {
    int count = fread(scratchCpu, sizeof(float), elements, f);
    if (count != elements) return 1;
    // copy and convert fp32->fp16
    cudaMemcpyAsync(scratchGpu, scratchCpu, sizeof(float) * elements, cudaMemcpyHostToDevice);
    ConvertFP32toFP16 <<<divUp(elements, 256), 256 >>> ((half*)w, (float*)scratchGpu, elements);
    return 0;
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

int checkpoint_init_weights(TransformerWeights* w, Config* p, FILE* f, int shared_weights) {
    size_t scratch_size = p->n_layers * std::max(p->dim, p->hidden_dim) * p->dim;
    scratch_size = std::max((size_t)p->vocab_size * p->dim, scratch_size);
    scratch_size *= sizeof(float);
    void* scratchCpu = malloc(scratch_size);
    void* scratchGpu = nullptr;
    cudaMalloc(&scratchGpu, scratch_size);
    if (uploadWeight(w->token_embedding_table, p->vocab_size * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->rms_att_weight, p->n_layers * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->wq, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->wk, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->wv, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->wo, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->rms_ffn_weight, p->n_layers * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->w1, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->w2, p->n_layers * p->hidden_dim * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->w3, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->rms_final_weight, p->dim, f, scratchCpu, scratchGpu)) return 1;

    int head_size = p->dim / p->n_heads;
    if (uploadWeight(w->freq_cis_real, p->seq_len * head_size / 2, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(w->freq_cis_imag, p->seq_len * head_size / 2, f, scratchCpu, scratchGpu)) return 1;

    if (!shared_weights)
        if (uploadWeight(w->wcls, p->vocab_size * p->dim, f, scratchCpu, scratchGpu)) return 1;

    cudaFree(scratchGpu);
    free(scratchCpu);
    return 0;
}

// ----------------------------------------------------------------------------
// neural net blocks

__global__ void elementwiseAdd(half* dest, half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dest[i] = (half)((float)dest[i] + (float)src[i]);
}

void accum(half* a, half* b, int size) {
    int blocks = divUp(size, 256);
    elementwiseAdd << <blocks, 256 >> > (a, b, size);
}


// Single block - not enough parallelism for the GPU, but it's just 1% of total time
__global__ void rmsNormKernel(half* o, half* x, half *weight, int size, int elementsPerThread)
{
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size)
            ss += (float) x[index];
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss * ss);

    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];
            o[index] = (half)val;
        }
    }
}

void rmsnorm(half* o, half* x, half* weight, int size) {
    int elementsPerThread = divUp(size, 1024);
    rmsNormKernel <<<1, 1024 >>> (o, x, weight, size, elementsPerThread);
}

// one output per warp so that we can parallelize the dot product across the warp
__global__ void matVecKernel(half* output, half* input, half* weight, int n, int d, int numSerialElements) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float) weight[index * n + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}


void matmul(half* xout, half* x, half* w, int n, int d) {
    int serialElements = divUp(n, 32);
    dim3 block_dim(32, 4);
    int blocks = divUp(d, 4);
    matVecKernel <<<blocks, block_dim >>> (xout, x, w, n, d, serialElements);
}

// Each block processes a single head
__global__ void RoPERotation_kernel(half* sq, half* sk, half* f_real, half* f_imag, int num_heads, int head_size) {
    int h = blockIdx.x;
    half* q = sq + h * head_size;
    half* k = sk + h * head_size;

    int i = threadIdx.x * 2;
    float q0 = q[i];
    float q1 = q[i + 1];
    float k0 = k[i];
    float k1 = k[i + 1];
    float fcr = f_real[i / 2];
    float fci = f_imag[i / 2];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;
    k[i] = k0 * fcr - k1 * fci;
    k[i + 1] = k0 * fci + k1 * fcr;
}

void RoPERotation(half *q, half *k, half *f_real, half *f_imag, int num_heads, int head_size) {
    RoPERotation_kernel <<<num_heads, head_size / 2 >>> (q, k, f_real, f_imag, num_heads, head_size);
}

__device__ void softmax_gpu(float* __restrict__ x, int size) {
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    int tid = threadIdx.x;
    int step = blockDim.x;

    // find max value (for numerical stability)
    float max_val = tid < size ? x[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (x[i] > max_val)
            max_val = x[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize
    for (int i = tid; i < size; i += step)
        x[i] /= sum;
}

// Each block processes a single head
// Poor parallelism and even poorer memory access pattern.
// Ankan - TODO: optimize this.
#define MAX_SEQ_LEN 8192
__global__ void MultiHeadAttention_kernel(half * __restrict__ output, const half* __restrict__ sq, 
    const half* __restrict__ key_cache, const half* __restrict__ value_cache,
    int num_heads, int head_size, int loff, int seq_len, int dim) {
    int h = blockIdx.x;

    // get the query vector for this head
    const half* q = sq + h * head_size;
    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];

    // iterate over all timesteps, including the current one
    for (int t = threadIdx.x; t < seq_len; t+= blockDim.x) {
        // get the key vector for this head and at this timestep
        const half* k = key_cache + loff + t * dim + h * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
            score += (float)q[i] * (float)k[i];
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }
    __syncthreads();

    // softmax the scores to get attention weights
    softmax_gpu(att, seq_len);
    __syncthreads();

    // weighted sum of the values, store back into xb
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++)
            val += att[t] * (float)value_cache[loff + t * dim + h * head_size + i];
        output[h * head_size + i] = (half) val;
    }
}

void MultiHeadAttention(half *output, half *q, half *key_cache, half *value_cache, int num_heads, int head_size, int loff, int seq_len) {
    int dim = head_size * num_heads;
    MultiHeadAttention_kernel <<<num_heads, 1024>>> (output, q, key_cache, value_cache, num_heads, head_size, loff, seq_len, dim);
}

__global__ void siluElementwiseMul_kernel(half* dest, half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = (float)dest[i];
        val *= 1.0f / (1.0f + expf(-val));
        val *= (float)src[i];
        dest[i] = (half)val;
    }
}

void siluElementwiseMul(half *hb, half *hb2, int size) {
    int blocks = divUp(size, 256);
    siluElementwiseMul_kernel <<<blocks, 256 >>> (hb, hb2, size);
}

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {

    // a few convenience variables
    half* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    half* content_row = &(w->token_embedding_table[token * dim]);
    cudaMemcpyAsync(x, content_row, dim * sizeof(half), cudaMemcpyDeviceToDevice);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    half* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    half* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l * dim * dim, dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        RoPERotation(s->q, s->k, freq_cis_real_row, freq_cis_imag_row, p->n_heads, head_size);

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        half* key_cache_row = s->key_cache + loff + pos * dim;
        half* value_cache_row = s->value_cache + loff + pos * dim;
        cudaMemcpyAsync(key_cache_row, s->k, dim * sizeof(half), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(value_cache_row, s->v, dim * sizeof(half), cudaMemcpyDeviceToDevice);

        MultiHeadAttention(s->xb, s->q, s->key_cache, s->value_cache, p->n_heads, head_size, loff, pos+1);

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // apply F.silu activation on hb and multiply it with hb2
        siluElementwiseMul(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);

    // copy logits from GPU->CPU
    ConvertFP16toFP32 <<<divUp(p->vocab_size, 256), 256 >>> (s->logits_temp, s->logits_gpu, p->vocab_size);
    cudaMemcpy(s->logits, s->logits_temp, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

int transformer_str(char* input, int pos, Config* p, RunState* s, TransformerWeights* w, char** vocab) {
    while (input[0] != 0) {
        int next = -1;
        int next_length = 0;
        for (int i = 0; i < p->vocab_size; i++) {
            char* v = vocab[i];
            int j = 0;
            int hit = 1;
            while (v[j] != 0) {
                if (0 == input[j] || v[j] != input[j]) {
                    hit = 0;
                    break;
                }
                ++j;
            }
            if (hit && j > next_length) {
                next_length = j;
                next = i;
            }
        }

        if (0 > next) return pos;

        pos++;
        transformer(next, pos, p, s, w);

        printf("[%s]", vocab[next]);
        fflush(stdout);

        input += next_length;
    }
    return pos;
}

int sample(float* probabilities, int n) {
    // sample index from probabilities, they must sum to 1
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int argmax(float* v, int n) {
    // return argmax of v in elements 0..n
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

// ----------------------------------------------------------------------------

long time_in_ms() {
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int main(int argc, char* argv[]) {

    // poor man's C argparse
    char* checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [temperature] [steps] [prompt]\n", argv[0]);
        return 1;
    }
    if (argc >= 2) {
        checkpoint = argv[1];
    }
    if (argc >= 3) {
        // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
        temperature = atof(argv[2]);
    }
    if (argc >= 4) {
        steps = atoi(argv[3]);
    }

    char* prompt = NULL;
    if (argc >= 5) {
        prompt = argv[4];
    }

    // seed rng with time. if you want deterministic behavior use temperature 0.0
    srand((unsigned int)time(NULL));

    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    int shared_weights;
    {
        FILE* file = fopen(checkpoint, "rb");
        if (!file) {
            printf("Unable to open the checkpoint file %s!\n", checkpoint);
            return 1;
        }
        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) { return 1; }

        // Dump model config
        printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\n\n",
            config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads, config.n_layers, config.seq_len, config.vocab_size);

        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // read in the Transformer weights
        malloc_weights(&weights, &config, shared_weights);
        if (checkpoint_init_weights(&weights, &config, file, shared_weights)) { return 1; }

    }
    // validate steps
    if (steps <= 0) { steps = config.seq_len; }

    // read in the tokenizer.bin file
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    {
        FILE* file = fopen("tokenizer.bin", "rb");
        if (!file) {
            printf("Unable to open the tokenizer file tokenizer.bin! Run "
                "python tokenizer.py to convert tokenizer.model -> tokenizer.bin\n");
            return 1;
        }
        int len;
        for (int i = 0; i < config.vocab_size; i++) {
            if (fread(&len, sizeof(int), 1, file) != 1) { return 1; }
            vocab[i] = (char*)malloc(len + 1);
            if (fread(vocab[i], len, 1, file) != 1) { return 1; }
            vocab[i][len] = '\0'; // add the string terminating token
        }
        fclose(file);
    }

    // create and init the application RunState
    RunState state;
    malloc_run_state(&state, &config);

    // the current position we are in
    long start = time_in_ms();
    int next;
    int token = 1; // 1 = BOS token in Llama-2 sentencepiece
    int pos = 0;
    printf("<s>\n"); // explicit print the initial BOS token (=1), stylistically symmetric
    transformer(token, pos, &config, &state, &weights);

    if (prompt) {
        pos = transformer_str(prompt, pos, &config, &state, &weights, vocab);
        steps += pos;
    }

    // can't run for more than seq_len total steps
    if (steps > config.seq_len)
        steps = config.seq_len;

    while (pos < steps) {
        // sample the next token
        if (temperature == 0.0f) {
            // greedy argmax sampling
            next = argmax(state.logits, config.vocab_size);
        }
        else {
            // apply the temperature to the logits
            for (int q = 0; q < config.vocab_size; q++) { state.logits[q] /= temperature; }
            
            // apply softmax to the logits to get the probabilities for next token
            softmax(state.logits, config.vocab_size);
            // we now want to sample from this distribution to get the next token
            next = sample(state.logits, config.vocab_size);
        }
        printf("%s", vocab[next]);
        fflush(stdout);

        // break if EOS token is reached
        if (next == 2)
            break;

        // advance forward
        token = next;
        pos++;

        // forward the transformer to get logits for the next token
        transformer(token, pos, &config, &state, &weights);
    }

    // report achieved tok/s
    long end = time_in_ms();
    double time = (end - start) / 1000.0;
    printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", pos / time, pos, time);

    // memory cleanup
    free_run_state(&state);
    free_weights(&weights, shared_weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}
