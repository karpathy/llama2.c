/* Inference for Llama-2 Transformer model in C + CUDA */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

// ----------------------------------------------------------------------------
// Transformer model

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
    // (optional) classifier weights for the logits, on the last layer
    uint8_t* wcls;
} TransformerWeights;

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
    // kv cache
    half* key_cache;   // (layer, seq_len, dim)
    half* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    uint8_t* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

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
    cudaMalloc((void**)&s->logits_gpu32, p->vocab_size * sizeof(float));
    cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(half));    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(half));

    // allocated on the CPU
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->att || !s->key_cache || !s->value_cache
     || !s->logits_gpu16 || !s->logits_gpu32 || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
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
    if (!shared_weights)
        cudaFree(w->wcls);
}

uint8_t* load_q8_weights(int num_layers, int layer_size, uint8_t** pptr) {
    int size = quant_size(num_layers, layer_size);
    uint8_t* w = NULL;
    cudaMalloc((void**)&w, size);
    if (!w) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    cudaMemcpyAsync(w, *pptr, size, cudaMemcpyHostToDevice);
    *pptr += size;
    return w;
}

void load_checkpoint_weights(TransformerWeights* w, Config* p, uint8_t* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;

    w->token_embedding_table = load_q8_weights(1, p->vocab_size * p->dim, &ptr);

    w->rms_att_weight = load_q8_weights(p->n_layers, p->dim, &ptr);

    w->wq = load_q8_weights(p->n_layers, p->dim * (p->n_heads * head_size), &ptr);
    w->wk = load_q8_weights(p->n_layers, p->dim * (p->n_kv_heads * head_size), &ptr);
    w->wv = load_q8_weights(p->n_layers, p->dim * (p->n_kv_heads * head_size), &ptr);
    w->wo = load_q8_weights(p->n_layers, (p->n_heads * head_size) * p->dim, &ptr);

    w->rms_ffn_weight = load_q8_weights(p->n_layers, p->dim, &ptr);

    w->w1 = load_q8_weights(p->n_layers, p->dim * p->hidden_dim, &ptr);
    w->w2 = load_q8_weights(p->n_layers, p->hidden_dim * p->dim, &ptr);
    w->w3 = load_q8_weights(p->n_layers, p->dim * p->hidden_dim, &ptr);

    w->rms_final_weight = load_q8_weights(1, p->dim, &ptr);

    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)

    if (shared_weights)
        w->wcls = w->token_embedding_table;
    else
        w->wcls = load_q8_weights(1, p->vocab_size * p->dim, &ptr);

}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, uint8_t** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (uint8_t*) mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    uint8_t* weights_ptr = *data + sizeof(Config);
    load_checkpoint_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

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
        float val = x[index];
        ss += (val * val);
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);

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
__global__ void RoPERotation_kernel(half* sq, half* sk, int pos, int num_heads, int num_kv_heads, int head_size) {
    int h = blockIdx.x;

    half* q = sq + h * head_size;
    half* k = sk + h * head_size;

    int i = threadIdx.x * 2;

    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    // rotate q
    float q0 = q[i];
    float q1 = q[i + 1];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    // rotate k
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
// neural net blocks; the dynamics of the Transformer

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

void RoPERotation(half *q, half *k, int pos, int num_heads, int num_kv_heads, int head_size) {
    RoPERotation_kernel <<< num_heads, head_size / 2 >>> (q, k, pos, num_heads, num_kv_heads, head_size);
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

void forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    half *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    //int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    dequantize_token(x, w->token_embedding_table, token, dim);

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

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(s->q, key_cache_row, pos, p->n_heads, p->n_kv_heads, head_size);

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
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    char byte_piece[2];
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
    t->byte_piece[1] = '\0'; // null terminate the byte_piece string
    t->sorted_vocab = NULL; // initialized lazily
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
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
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        // ok this token is a raw byte token, careful to only print printable chars or whitespace
        // some of the other bytes can be various control codes, backspace, etc. => skip
        if (isprint(byte_val) || isspace(byte_val)) {
            t->byte_piece[0] = byte_val;
            piece = &t->byte_piece[0];
        }
    }
    return piece;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array

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
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_lenght is 1)
    char* str_buffer = (char*) malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // add_dummy_prefix is true by default
    tokens[0] = str_lookup(" ", t->sorted_vocab, t->vocab_size);
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

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
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
        // apply the temperature to the logits
        float inv_temperature = 1.0f / sampler->temperature;
        scalar_mul32_kernel <<< divUp(sampler->vocab_size, 256), 256 >>> (state->logits_gpu32, inv_temperature, sampler->vocab_size);
        // apply softmax to the logits to get the probabilities for next token
        softmax32_kernel <<< 1, 1024 >>> (state->logits_gpu32, sampler->vocab_size);
        // copy the logits from GPU to the CPU
        cudaMemcpy(state->logits, state->logits_gpu32, sampler->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
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

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {

    // encode the (string) prompt into tokens sequence, if any is given
    int *prompt_tokens = NULL; // the sequence of prompt tokens
    int num_prompt_tokens = 0; // the total number of prompt tokens
    if (prompt != NULL) {
        prompt_tokens = (int*)malloc((strlen(prompt)+1) * sizeof(int));
        encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        forward(transformer, token, pos);

        // advance the state state machine
        if (pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, &transformer->state);
        }
        pos++;

        // data-dependent terminating condition: the BOS (1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        printf("%s", piece);
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

    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// int main

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
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;        // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;          // number of steps to run for
    char *prompt = NULL;      // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

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
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0) steps = transformer.config.seq_len; // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
