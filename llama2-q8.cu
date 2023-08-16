/*
Inference for Llama-2 Transformer model in C + CUDA.
*/

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// ----------------------------------------------------------------------------
// Transformer and RunState structs

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
    float base;
    float scale;
    uint8_t weights[];
} q8data;

#define quant_size(n_layers,size) (n_layers * (4 + 4 + size))

typedef struct {
    // token embedding table
    uint8_t* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    uint8_t* rms_att_weight; // (layer, dim) rmsnorm weights
    uint8_t* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    uint8_t* wq; // (layer, dim, n_heads * head_size)
    uint8_t* wk; // (layer, dim, n_kv_heads * head_size)
    uint8_t* wv; // (layer, dim, n_kv_heads * head_size)
    uint8_t* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    uint8_t* w1; // (layer, hidden_dim, dim)
    uint8_t* w2; // (layer, dim, hidden_dim)
    uint8_t* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    uint8_t* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    half* freq_cis_real; // (seq_len, head_size/2)
    half* freq_cis_imag; // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    uint8_t* wcls;
} TransformerWeights;

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    // current wave of activations
    half* x; // activation at current time stamp (dim,)
    half* xb; // same, but inside a residual branch (dim,)
    half* xb2; // an additional buffer just for convenience (dim,)
    half* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* q; // query (dim,)
    half* att; // buffer for scores/attention values (n_heads, seq_len)
    half* logits_gpu16; // output logits
    float* logits_gpu32; // logits in GPU memory converted to float
    float* logits; // logits copied CPU side
    ProbIndex *probindex; // buffer used in top-p sampling
    // kv cache
    half* key_cache;   // (layer, seq_len, dim)
    half* value_cache; // (layer, seq_len, dim)
} RunState;

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void scalar_mul32_kernel(float* arr, float value, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        arr[i] = arr[i] * value;
}

__global__ void element_wise_add_kernel(half* dest, half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dest[i] = (half)((float)dest[i] + (float)src[i]);
}

__global__ void convert_fp32_to_fp16(half* out, float* in, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        out[index] = (half)in[index];
}

__global__ void convert_fp16_to_fp32(float* out, half* in, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        out[index] = (float)in[index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
__global__ void rmsnorm_kernel(half* o, half* x, q8data* q8, int size) {
    float ss = 0.0f;
    for (int index = threadIdx.x; index < size; index+=1024) {
        ss += (float)x[index];
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

    float base = q8->base;
    float scale = q8->scale;
    uint8_t *qweight = q8->weights;

    // normalize
    for (int index = threadIdx.x; index < size; index+=1024) {
        float val = ((float)x[index]) * ss * (base + scale * qweight[index]);
        o[index] = (half)val;
    }
}

// Note that ~95% of total time is spent here, so optimizing this is important
// One output generated per warp so that we can parallelize the dot product across the warp
__global__ void mat_vec_q8_kernel(half* output, const half* __restrict__ input, q8data* __restrict__ q8,
    int n, int d, int numSerialElements,
    int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha) {

    float base = q8->base;
    float scale = q8->scale;
    uint8_t *weight = q8->weights;

    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    input  += blockIdx.y * input_stride;
    weight += blockIdx.y * weight_stride + index * weight_row_stride;
    output += blockIdx.y * output_stride;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n) {
            // dequantize the weight, multiply and accumulate
            sum += (base + scale * weight[j]) * (float)input[j];
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

__global__ void mat_vec_kernel(half* output, const half* __restrict__ input, const half* __restrict__ weight,
    int n, int d, int numSerialElements,
    int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha) {

    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    input  += blockIdx.y * input_stride;
    weight += blockIdx.y * weight_stride + index * weight_row_stride;
    output += blockIdx.y * output_stride;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
__global__ void vec_mat_kernel(half* output, const half* __restrict__ input, const half* __restrict__ weight,
    int N, int K, int elementsPerThread,
    int input_stride, int weight_stride, int output_stride, int weight_row_stride) {

    input  += blockIdx.y * input_stride;
    weight += blockIdx.y * weight_stride;
    output += blockIdx.y * output_stride;

    int start_n = blockIdx.x * 32;
    int i = start_n + threadIdx.y;

    // 2x for double buffering
    // +2 to avoid shared memory bank conflicts
    __shared__ half loaded_fragment[2][32][32 + 2];

    // OOB check
    if (i >= N)
        return;

    // load the first 32x32 fragment
    int n = start_n + threadIdx.x;
    int k = threadIdx.y;
    int offset = k * weight_row_stride + n;
    loaded_fragment[0][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0;

    float sum = 0;
    // Loop over the matrix row and vector elements
    for (int e = 0; e < elementsPerThread;)
    {
        __syncthreads();    // wait for the load

        int start_k = e * 32;
        k = start_k + threadIdx.x;
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][threadIdx.x][threadIdx.y]) * (float)(input[k]);

        // load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + threadIdx.x;
        k = start_k + threadIdx.y;
        int offset = k * weight_row_stride + n;
        loaded_fragment[buf_i][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0;
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[i] = (half)sum;
}

// Each block processes a single head
__global__ void RoPERotation_kernel(half* sq, half* sk, half* f_real, half* f_imag, int num_heads, int num_kv_heads, int head_size) {
    int h = blockIdx.x;

    half* q = sq + h * head_size;
    half* k = sk + h * head_size;

    int i = threadIdx.x * 2;
    int j = threadIdx.x;

    float fcr = f_real[j];
    float fci = f_imag[j];

    float q0 = q[i];
    float q1 = q[i + 1];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    if (h < num_kv_heads) {
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i] = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

#define MAX_SEQ_LEN 8192
__global__ void softmax_kernel(half* __restrict__ arr, int num_heads, int size) {
    __shared__ float att[MAX_SEQ_LEN];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;

    half* __restrict__ arr_base = arr + h * size;

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float) arr_base[t];
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? att[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (att[i] > max_val)
            max_val = att[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        att[i] = expf(att[i] - max_val);
        sum += att[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    float inv_sum = 1.0f / sum;
    for (int t = tid; t < size; t += step)
        arr_base[t] = (half) (att[t] * inv_sum);
}

__global__ void softmax32_kernel(float* __restrict__ x, int size) {
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

__global__ void argmax32_kernel(float* __restrict__ x, int size, int *result) {
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    int tid = threadIdx.x;
    int step = blockDim.x;

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
    float global_max_val;
    global_max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = global_max_val;
    __syncthreads();
    global_max_val = shared_val;

    // get its position
    if (max_val == global_max_val) {
        *result = max_pos;
    }
}

__global__ void silu_element_wise_mul_kernel(half* dest, half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = (float)dest[i];
        val *= 1.0f / (1.0f + expf(-val));
        val *= (float)src[i];
        dest[i] = (half)val;
    }
}

__global__ void dequantize_token_kernel(half* x, q8data* q8, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        uint8_t *qweight = q8->weights + token * dim;
        x[i] = (half) (q8->base + q8->scale * qweight[i]);
    }
}

// ----------------------------------------------------------------------------
// Memory management functions

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    // allocated on the GPU
    cudaMalloc((void**)&s->x, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb2, p->dim * sizeof(half));
    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->q, p->dim * sizeof(half));
    cudaMalloc((void**)&s->att, p->n_heads * p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits_gpu16, p->vocab_size * sizeof(half));
    cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(half));    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(half));
    cudaMalloc((void**)&s->logits_gpu32, p->vocab_size * sizeof(float));

    // allocated on the CPU
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));
    s->probindex = (ProbIndex*)calloc(p->vocab_size, sizeof(ProbIndex));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->att || !s->key_cache || !s->value_cache
        || !s->logits_gpu16 || !s->logits_gpu32 || !s->logits || !s->probindex) {
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
    cudaFree(s->att);
    cudaFree(s->logits_gpu16);
    cudaFree(s->logits_gpu32);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
    free(s->logits);
    free(s->probindex);
}

void malloc_weights(TransformerWeights* w, Config* p, int shared_weights) {
    int head_size = p->dim / p->n_heads;

    cudaMalloc((void**)&w->token_embedding_table, quant_size(1, p->vocab_size * p->dim));
    cudaMalloc((void**)&w->rms_att_weight, quant_size(p->n_layers, p->dim));
    cudaMalloc((void**)&w->rms_ffn_weight, quant_size(p->n_layers, p->dim));
    cudaMalloc((void**)&w->wq, quant_size(p->n_layers, p->dim * (p->n_heads * head_size)));
    cudaMalloc((void**)&w->wk, quant_size(p->n_layers, p->dim * (p->n_kv_heads * head_size)));
    cudaMalloc((void**)&w->wv, quant_size(p->n_layers, p->dim * (p->n_kv_heads * head_size)));
    cudaMalloc((void**)&w->wo, quant_size(p->n_layers, (p->n_heads * head_size) * p->dim));
    cudaMalloc((void**)&w->w1, quant_size(p->n_layers, p->dim * p->hidden_dim));
    cudaMalloc((void**)&w->w2, quant_size(p->n_layers, p->hidden_dim * p->dim));
    cudaMalloc((void**)&w->w3, quant_size(p->n_layers, p->dim * p->hidden_dim));
    cudaMalloc((void**)&w->rms_final_weight, quant_size(1, p->dim));

    cudaMalloc((void**)&w->freq_cis_real, p->seq_len * head_size / 2 * sizeof(half));
    cudaMalloc((void**)&w->freq_cis_imag, p->seq_len * head_size / 2 * sizeof(half));

    if (shared_weights)
        w->wcls = w->token_embedding_table;
    else
        cudaMalloc((void**)&w->wcls, quant_size(1, p->vocab_size * p->dim));

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

int load_q8_weights(void *w, int size, FILE* f, void *scratchCPU) {
    int count = fread(scratchCPU, sizeof(int8_t), size, f);
    if (count != size) return 1;
    cudaMemcpyAsync(w, scratchCPU, size, cudaMemcpyHostToDevice);
    return 0;
}

int load_weights(void *w, int elements, FILE* f, void *scratchCPU, void *scratchGPU) {
    int count = fread(scratchCPU, sizeof(float), elements, f);
    if (count != elements) return 1;
    // copy and convert fp32->fp16
    cudaMemcpyAsync(scratchGPU, scratchCPU, sizeof(float) * elements, cudaMemcpyHostToDevice);
    convert_fp32_to_fp16 <<< divUp(elements, 1024), 1024 >>> ((half*)w, (float*)scratchGPU, elements);
    return 0;
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

int checkpoint_init_weights(TransformerWeights* w, Config* p, FILE* f, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    size_t scratch_size = p->n_layers * std::max(p->dim, p->hidden_dim) * p->dim;
    scratch_size = std::max((size_t)p->vocab_size * p->dim, scratch_size);
    scratch_size *= sizeof(float);
    void* scratchCPU = malloc(scratch_size);
    void* scratchGPU = nullptr;
    cudaMalloc(&scratchGPU, scratch_size);

    if (load_q8_weights(w->token_embedding_table, quant_size(1, p->vocab_size * p->dim), f, scratchCPU)) return 1;
    if (load_q8_weights(w->rms_att_weight, quant_size(p->n_layers, p->dim), f, scratchCPU)) return 1;
    if (load_q8_weights(w->wq, quant_size(p->n_layers, p->dim * (p->n_heads * head_size)), f, scratchCPU)) return 1;
    if (load_q8_weights(w->wk, quant_size(p->n_layers, p->dim * (p->n_kv_heads * head_size)), f, scratchCPU)) return 1;
    if (load_q8_weights(w->wv, quant_size(p->n_layers, p->dim * (p->n_kv_heads * head_size)), f, scratchCPU)) return 1;
    if (load_q8_weights(w->wo, quant_size(p->n_layers, (p->n_heads * head_size) * p->dim), f, scratchCPU)) return 1;
    if (load_q8_weights(w->rms_ffn_weight, quant_size(p->n_layers, p->dim), f, scratchCPU)) return 1;
    if (load_q8_weights(w->w1, quant_size(p->n_layers, p->dim * p->hidden_dim), f, scratchCPU)) return 1;
    if (load_q8_weights(w->w2, quant_size(p->n_layers, p->hidden_dim * p->dim), f, scratchCPU)) return 1;
    if (load_q8_weights(w->w3, quant_size(p->n_layers, p->dim * p->hidden_dim), f, scratchCPU)) return 1;
    if (load_q8_weights(w->rms_final_weight, quant_size(1, p->dim), f, scratchCPU)) return 1;

    if (load_weights(w->freq_cis_real, p->seq_len * head_size / 2, f, scratchCPU, scratchGPU)) return 1;
    if (load_weights(w->freq_cis_imag, p->seq_len * head_size / 2, f, scratchCPU, scratchGPU)) return 1;

    if (!shared_weights)
        if (load_q8_weights(w->wcls, quant_size(1, p->vocab_size * p->dim), f, scratchCPU)) return 1;

    cudaFree(scratchGPU);
    free(scratchCPU);
    return 0;
}

// ----------------------------------------------------------------------------
// neural net blocks

void accum(half* a, half* b, int size) {
    int blocks = divUp(size, 1024);
    element_wise_add_kernel <<< blocks, 1024 >>> (a, b, size);
}

void rmsnorm(half* o, half* x, uint8_t *wptr, int l, int size) {
    q8data* q8 = (q8data*) (wptr + quant_size(l, size));
    rmsnorm_kernel <<< 1, 1024 >>> (o, x, q8, size);
}

void matmul(half* xout, half* x, uint8_t *wptr, int l, int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    q8data* q8 = (q8data*) (wptr + quant_size(l, n * d));
    int serialElements = divUp(n, 32);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(d, 4), batch);
    if (w_row_stride == -1) w_row_stride = n;

    mat_vec_q8_kernel <<< grid_dim, block_dim >>> (xout, x, q8, n, d, serialElements, x_stride, w_stride, op_stride, w_row_stride, alpha);
}

void RoPERotation(half *q, half *k, half *f_real, half *f_imag, int num_heads, int num_kv_heads, int head_size) {
    RoPERotation_kernel <<< num_heads, head_size / 2 >>> (q, k, f_real, f_imag, num_heads, num_kv_heads, head_size);
}

void MultiHeadAttention(half *output, half *q, half *key_cache, half *value_cache, half *att, int num_heads, int head_size, int seq_len) {
    int dim = head_size * num_heads;

    // 1. Get attention scores
    int serialElements = divUp(head_size, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim1(divUp(seq_len, 32), num_heads);
    mat_vec_kernel <<< grid_dim1, block_dim >>> (att, q, key_cache, head_size, seq_len, serialElements, head_size, head_size, seq_len, dim, 1.0 / sqrt(head_size));

    // 2. Run softmax kernel
    softmax_kernel <<< num_heads, 1024 >>> (att, num_heads, seq_len);

    // 3. weighted sum of the values to get the final result
    serialElements = divUp(seq_len, 32);    
    dim3 grid_dim2(divUp(head_size, 32), num_heads);
    vec_mat_kernel <<< grid_dim2, block_dim >>> (output, att, value_cache, head_size, seq_len, serialElements, seq_len, head_size, head_size, dim);
}

void siluElementwiseMul(half *hb, half *hb2, int size) {
   silu_element_wise_mul_kernel <<< divUp(size, 1024), 1024 >>> (hb, hb2, size);
}

void dequantize_token(half* x, uint8_t* wptr, int token, int dim) {
    q8data* q8 = (q8data*) wptr;
    dequantize_token_kernel <<<divUp(dim, 1024), 1024>>> (x, q8, token, dim);
}

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {

    // a few convenience variables
    half* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    //int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    dequantize_token(x, w->token_embedding_table, token, dim);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    half* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    half* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight, l, dim);

        // we directly store (key, value) at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        half* key_cache_row = s->key_cache + loff + pos * kv_dim;
        half* value_cache_row = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq, l, dim, dim);
        matmul(key_cache_row, s->xb, w->wk, l, dim, kv_dim);
        matmul(value_cache_row, s->xb, w->wv, l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(s->q, key_cache_row, freq_cis_real_row, freq_cis_imag_row, p->n_heads, p->n_kv_heads, head_size);

        // apply MHA using the query and the key-value cache
        MultiHeadAttention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, p->n_heads, head_size, pos+1);

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo, l, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight, l, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1, l, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3, l, dim, hidden_dim);

        // apply F.silu activation on hb and multiply it with hb2
        siluElementwiseMul(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2, l, hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, 0, dim);

    // classifier into logits
    matmul(s->logits_gpu16, x, w->wcls, 0, p->dim, p->vocab_size);

    // copy logits from GPU->CPU
    convert_fp16_to_fp32 <<<divUp(p->vocab_size, 1024), 1024 >>> (s->logits_gpu32, s->logits_gpu16, p->vocab_size);
}

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

typedef struct {
    char *str;
    int id;
} TokenIndex;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void bpe_encode(char *text, char **vocab, float *vocab_scores, int vocab_size, unsigned int max_token_length, int *tokens, int *n_tokens) {

    // sort vocabulary
    TokenIndex *sorted_vocab = (TokenIndex*) malloc(vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < vocab_size; i++) {
        sorted_vocab[i].str = vocab[i];
        sorted_vocab[i].id = i;
    }
    qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    char* str_buffer = (char*) malloc((max_token_length*2 +1 +2) * sizeof(char)); // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_lenght is 1)
    size_t str_len = 0;

    // add_dummy_prefix is true by default
    tokens[0] = str_lookup(" ", sorted_vocab, vocab_size);
    *n_tokens = 1; // the number of tokens

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
        int id = str_lookup(str_buffer, sorted_vocab, vocab_size);

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
            sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
            if (id != -1 && vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores[id];
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

    free(str_buffer);
    free(sorted_vocab);
}

// ----------------------------------------------------------------------------
// utilities: time / rng

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

unsigned long long rng_seed;
unsigned int random_u32() {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32() { // random float32 in [0,1)
    return (random_u32() >> 8) / 16777216.0f;
}

// ----------------------------------------------------------------------------
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

int argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_pos;
    int *pmax_pos;

    // allocate memory on the device
    cudaMalloc((void**)&pmax_pos, sizeof(int));

    // call the kernel
    argmax32_kernel<<<1,1024>>>(probabilities, n, pmax_pos);

    // copy the result back to host
    cudaMemcpy(&max_pos, pmax_pos, sizeof(int), cudaMemcpyDeviceToHost);

    // free the allocated memory
    cudaFree(pmax_pos);

    return max_pos;
}

int sample(float* probabilities, int n) {
    // sample index from probabilities (they must sum to 1!)
    float r = random_f32();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
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

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".

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
    float r = random_f32() * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}


// ----------------------------------------------------------------------------
// int main

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature, default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling. default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default inits
    char *checkpoint = NULL;  // e.g. out/model.bin
    char *tokenizer = "tokenizer.bin";
    float temperature = 1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;        // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    rng_seed = 0; // seed rng with time by default
    int steps = 256;          // number of steps to run for
    char *prompt = NULL;      // prompt string

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint = argv[1]; } else { error_usage(); }
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
        else if (argv[i][1] == 'z') { tokenizer = argv[i + 1]; }
        else { error_usage(); }
    }
    if(rng_seed == 0) { rng_seed =  (unsigned int)time(NULL);}

    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    int shared_weights;
    {
        FILE *file = fopen(checkpoint, "rb");
        if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); return 1; }
        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // read in the Transformer weights
        malloc_weights(&weights, &config, shared_weights);
        if (checkpoint_init_weights(&weights, &config, file, shared_weights)) { return 1; }
    }
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }

    // read in the tokenizer .bin file
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    float* vocab_scores = (float*)malloc(config.vocab_size * sizeof(float));
    unsigned int max_token_length;
    {
        FILE *file = fopen(tokenizer, "rb");
        if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer); return 1; }
        if (fread(&max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1; }
        int len;
        for (int i = 0; i < config.vocab_size; i++) {
            if (fread(vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1;}
            if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1; }
            vocab[i] = (char *)malloc(len + 1);
            if (fread(vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1; }
            vocab[i][len] = '\0'; // add the string terminating token
        }
        fclose(file);
    }

    // create and init the application RunState
    RunState state;
    malloc_run_state(&state, &config);

    // process the prompt, if any
    int *prompt_tokens = NULL;
    int num_prompt_tokens = 0;
    if (prompt != NULL) {
        prompt_tokens = (int*)malloc((strlen(prompt)+1) * sizeof(int));
        bpe_encode(prompt, vocab, vocab_scores, config.vocab_size, max_token_length, prompt_tokens, &num_prompt_tokens);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        transformer(token, pos, &config, &state, &weights);

        // advance the state state machine
        if(pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        } else {
            // sample the next token
            if (temperature == 0.0f) {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits_gpu32, config.vocab_size);
            } else {
                // apply the temperature to the logits
                float inv_temperature = 1.0f / temperature;
                scalar_mul32_kernel <<< divUp(config.vocab_size, 256), 256 >>> (state.logits_gpu32, inv_temperature, config.vocab_size);
                // apply softmax to the logits to get the probabilities for next token
                softmax32_kernel <<< 1, 1024 >>> (state.logits_gpu32, config.vocab_size);
                // copy the logits from GPU to the CPU
                cudaMemcpy(state.logits, state.logits_gpu32, config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
                // we sample from this distribution to get the next token
                if (topp <= 0 || topp >= 1) {
                    // simply sample from the predicted probability distribution
                    next = sample(state.logits, config.vocab_size);
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    next = sample_topp(state.logits, config.vocab_size, topp, state.probindex);
                }
            }
        }
        pos++;

        // data-dependent terminating condition: the BOS (1) token delimits sequences
        if (next == 1) { break; }

        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        char *token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next]+1 : vocab[next];
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        unsigned char byte_val;
        if (sscanf(token_str, "<0x%02hhX>", &byte_val) == 1) {
            // ok this token is a raw byte token, carefuly to only print printable chars or whitespace
            // some of the other bytes can be various control codes, backspace, etc. => skip
            if (isprint(byte_val) || isspace(byte_val)) {
                char byte_piece[2];
                byte_piece[0] = byte_val;
                byte_piece[1] = '\0';
                printf("%s", byte_piece);
            }
        } else {
            printf("%s", token_str);
        }
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    // memory and file handles cleanup
    free_run_state(&state);
    free_weights(&weights, shared_weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    free(vocab_scores);
    if (prompt_tokens != NULL) free(prompt_tokens);
    return 0;
}
