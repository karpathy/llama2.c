/*
Inference for Llama-2 Transformer model in pure C.

Example compile: (see README for more details)
$ gcc -O3 -o run run.c -lm

Then run with:
$ ./run
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>  // for uintptr_t
#ifdef LLAMAC_AVX2
#include <immintrin.h> // AVX2
#endif

// Processor memory alignment stride
#define CACHE_ALIGN_SIZE 32

void aligned_assignment_helper(void* pointer, char* pointer_name, int line_info) {
    int is_aligned = ((uintptr_t)pointer) % CACHE_ALIGN_SIZE;
    if (is_aligned != 0) {
        printf("Bad Pointer Alignement of Variable %s (%p) at line %d\r\n", pointer_name, (void*) pointer, line_info);
        // File Config header must be aligned to CACHE_ALIGN_SIZE for AVX2 to work
        printf("Did you forget to upgrade to the latest file format\r\n");
        exit(1);
    }
}

// #define CHECK_ALIGNMENT(pointer) aligned_assignment_helper(pointer, #pointer, __LINE__)
#define CHECK_ALIGNMENT(pointer)

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

// Config structure needs to be CACHE ALIGNED (Typically 32 Bytes)
// If you change this, it is important that export_meta_llama_bin.py is updated as well
typedef struct {
    int magic_and_version; // header version and struct alignment 0x42 0xMajor 0xMinor 0xPoint
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
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

void* aligned_calloc(size_t nitems, size_t size) {
    void* mem = NULL;
    size_t block_size = nitems * size;
    if (posix_memalign(&mem, CACHE_ALIGN_SIZE, block_size)) {
        return NULL;
    }
    return mem;
}

void malloc_run_state(RunState* s, Config* p) {
    // we aligned_calloc instead of malloc to keep valgrind happy
    s->x = aligned_calloc(p->dim, sizeof(float));
    s->xb = aligned_calloc(p->dim, sizeof(float));
    s->xb2 = aligned_calloc(p->dim, sizeof(float));
    s->hb = aligned_calloc(p->hidden_dim, sizeof(float));
    s->hb2 = aligned_calloc(p->hidden_dim, sizeof(float));
    s->q = aligned_calloc(p->dim, sizeof(float));
    s->k = aligned_calloc(p->dim, sizeof(float));
    s->v = aligned_calloc(p->dim, sizeof(float));
    s->att = aligned_calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = aligned_calloc(p->vocab_size, sizeof(float));
    s->key_cache = aligned_calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->value_cache = aligned_calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q 
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache 
     || !s->value_cache) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

void checkpoint_init_weights(TransformerWeights *w, Config* p, float* f, int shared_weights) {
    CHECK_ALIGNMENT(f);

    float* ptr = f;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->wq = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wk = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wv = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wo = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->w1 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->freq_cis_real = ptr;
    int head_size = p->dim / p->n_heads;
    ptr += p->seq_len * head_size / 2;
    w->freq_cis_imag = ptr;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

// ----------------------------------------------------------------------------
// neural net blocks

void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

#ifndef LLAMAC_AVX2
// straight C implementation
void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
#else
// rmsnorm is a function that normalizes a given vector (x) using Root Mean Square (RMS)
// normalization and then scales it by a weight vector. The result is stored in the output
// vector (o). The size of the vectors is given by the input parameter 'size'.
// RMS normalization is used to prevent the exploding gradient problem in deep learning
// models, improving the stability of training.
void rmsnorm(float* o, const float* x, const float* weight, int size) {
    // Calculate sum of squares
    int n = size / 8 * 8;  // make size multiple of 8
    __m256 ss_vec = _mm256_setzero_ps();  // initialize with 0s
    for (int j = 0; j < n; j += 8) {
        __m256 x_vec = _mm256_load_ps(&x[j]); // load 8 x values
        ss_vec = _mm256_fmadd_ps(x_vec, x_vec, ss_vec); // fused multiply-add
    }
    // Handle tail part
    float ss = 0.0f;
    for (int j = n; j < size; j++) {
        ss += x[j] * x[j];
    }
    // Horizontal add
    ss_vec = _mm256_hadd_ps(ss_vec, ss_vec);
    ss_vec = _mm256_hadd_ps(ss_vec, ss_vec);
    float ss_vals[8];
    _mm256_store_ps(ss_vals, ss_vec);
    ss += ss_vals[0] + ss_vals[4];

    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    // Normalize and scale
    __m256 ss_vec_norm = _mm256_set1_ps(ss); // broadcast ss to all elements of the vector
    for (int j = 0; j < n; j += 8) {
        __m256 x_vec = _mm256_load_ps(&x[j]); // load 8 x values
        __m256 w_vec = _mm256_load_ps(&weight[j]); // load 8 weight values

        // Perform the weight * (ss * x) operation
        __m256 o_vec = _mm256_mul_ps(x_vec, ss_vec_norm); // perform ss * x
        o_vec = _mm256_mul_ps(o_vec, w_vec); // multiply with weight

        // Store the result
        _mm256_store_ps(&o[j], o_vec);
    }
    // Handle tail part
    for (int j = n; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
#endif

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

    float inv_sum = 1.0f / sum;

    // normalize
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

#ifndef LLAMAC_AVX2
// straight C implementation
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int i_n = i * n;
        for (int j = 0; j < n; j++) {
            val += w[i_n + j] * x[j];
        }
        xout[i] = val;
    }
}
#else
// matmul is a matrix multiplication function that computes the product of a matrix (w) and
// a vector (x) and stores the result in the output vector (o).
// The function takes the dimensions (n, d) of the matrix as input.
// This function is used for transforming input data in the application
void matmul(float* o, const float* x, const float* w, int n, int d) {
    CHECK_ALIGNMENT(o);
    CHECK_ALIGNMENT(x);
    CHECK_ALIGNMENT(w);

    // W (d,n) @ x (n,) -> o (d,)
    int nn = n / 8 * 8;  // ensure n is a multiple of 8
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        __m256 sum_vec = _mm256_setzero_ps(); // for AVX2, sum of 8 floats
        int i_n = i * n;
        for (int j = 0; j < nn; j += 8) {
            // Load 8 values from w and x
            __m256 w_vec = _mm256_load_ps(&w[i_n + j]);
            __m256 x_vec = _mm256_load_ps(&x[j]);

            // Multiply and accumulate
            __m256 prod_vec = _mm256_mul_ps(w_vec, x_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod_vec);
        }

        // Perform horizontal add
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        float vals[8];
        _mm256_storeu_ps(vals, sum_vec);
        float val = vals[0] + vals[4];

        // handle remainder if n is not a multiple of 8
        for (int j = nn; j < n; j++) {
            val += w[i_n + j] * x[j];
        }
        o[i] = val;
    }
}

// matmul2 is an optimized matrix multiplication function which computes two matrix-vector
// products in one function. The input matrices are w1 and w2, and the vector is x.
// The results are stored in o1 and o2 respectively.
// This function aims to improve the memory locality and cache utilization by fusing
// two similar computations, which potentially results in performance improvements.
void matmul2(float* o1, float* o2, const float* x, const float* w1, const float* w2, int n, int d) {
    CHECK_ALIGNMENT(o1);
    CHECK_ALIGNMENT(o2);
    CHECK_ALIGNMENT(x);
    CHECK_ALIGNMENT(w1);
    CHECK_ALIGNMENT(w2);

    // o1 = W1 @ x
    // o2 = W3 @ x
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        __m256 o1_val_vec = _mm256_setzero_ps();
        __m256 o2_val_vec = _mm256_setzero_ps();
        int i_n = i * n;
        for (int j = 0; j < n; j += 8) {
            __m256 x_vec = _mm256_load_ps(&x[j]);

            o1_val_vec = _mm256_fmadd_ps(_mm256_load_ps(&w1[i_n + j]), x_vec, o1_val_vec);
            o2_val_vec = _mm256_fmadd_ps(_mm256_load_ps(&w2[i_n + j]), x_vec, o2_val_vec);
        }

        // Perform horizontal add
        o1_val_vec = _mm256_hadd_ps(o1_val_vec, o1_val_vec);
        o1_val_vec = _mm256_hadd_ps(o1_val_vec, o1_val_vec);
        o2_val_vec = _mm256_hadd_ps(o2_val_vec, o2_val_vec);
        o2_val_vec = _mm256_hadd_ps(o2_val_vec, o2_val_vec);

        float o1_vals[8], o2_vals[8];

        _mm256_storeu_ps(o1_vals, o1_val_vec);
        _mm256_storeu_ps(o2_vals, o2_val_vec);

        o1[i] = o1_vals[0] + o1_vals[4];
        o2[i] = o2_vals[0] + o2_vals[4];
    }
}

// matmul3 is an optimized matrix multiplication function which computes three matrix-vector
// products in one function. The input matrices are w1, w2, and w3, and the vector is x.
// The results are stored in o1, o2, and o3 respectively.
// This function aims to improve the memory locality and cache utilization by fusing
// three similar computations, which potentially results in performance improvements.
void matmul3(float* o1, float* o2, float* o3, const float* x, const float* w1, const float* w2, const float* w3, int n, int d) {
    CHECK_ALIGNMENT(o1);
    CHECK_ALIGNMENT(o2);
    CHECK_ALIGNMENT(o3);
    CHECK_ALIGNMENT(x);
    CHECK_ALIGNMENT(w1);
    CHECK_ALIGNMENT(w2);
    CHECK_ALIGNMENT(w3);

    // o1 = W1 @ x
    // o2 = W2 @ x
    // o3 = W3 @ x
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        __m256 o1_val_vec = _mm256_setzero_ps();
        __m256 o2_val_vec = _mm256_setzero_ps();
        __m256 o3_val_vec = _mm256_setzero_ps();
        int i_n = i * n;
        for (int j = 0; j < n; j += 8) {
            __m256 x_vec = _mm256_load_ps(&x[j]);

            o1_val_vec = _mm256_fmadd_ps(_mm256_load_ps(&w1[i_n + j]), x_vec, o1_val_vec);
            o2_val_vec = _mm256_fmadd_ps(_mm256_load_ps(&w2[i_n + j]), x_vec, o2_val_vec);
            o3_val_vec = _mm256_fmadd_ps(_mm256_load_ps(&w3[i_n + j]), x_vec, o3_val_vec);
        }

        // Perform horizontal add
        o1_val_vec = _mm256_hadd_ps(o1_val_vec, o1_val_vec);
        o1_val_vec = _mm256_hadd_ps(o1_val_vec, o1_val_vec);
        o2_val_vec = _mm256_hadd_ps(o2_val_vec, o2_val_vec);
        o2_val_vec = _mm256_hadd_ps(o2_val_vec, o2_val_vec);
        o3_val_vec = _mm256_hadd_ps(o3_val_vec, o3_val_vec);
        o3_val_vec = _mm256_hadd_ps(o3_val_vec, o3_val_vec);

        float o1_vals[8], o2_vals[8], o3_vals[8];

        _mm256_storeu_ps(o1_vals, o1_val_vec);
        _mm256_storeu_ps(o2_vals, o2_val_vec);
        _mm256_storeu_ps(o3_vals, o3_val_vec);

        o1[i] = o1_vals[0] + o1_vals[4];
        o2[i] = o2_vals[0] + o2_vals[4];
        o3[i] = o3_vals[0] + o3_vals[4];
    }
}
#endif

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {
    
    // a few convenience variables
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    float inv_sqrt_head_size = 1.0f / sqrtf(head_size);

    // copy the token embedding into x
    float* content_row = &(w->token_embedding_table[token * dim]);
    memcpy(x, content_row, dim*sizeof(*x));

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {
    
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
#ifndef LLAMAC_AVX2
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);
#else
        // used fused / loop-jamming version when we have AVX2
        matmul3(s->q, s->k, s->v, s->xb, w->wq + l*dim*dim, w->wk + l*dim*dim, w->wv + l*dim*dim, dim, dim);
#endif

        // apply RoPE rotation to the q and k vectors for each head
        for (int h = 0; h < p->n_heads; h++) {
            // common expression hoisting
            int h_head_size = h * head_size;

            // get the q and k vectors for this head
            float* q = s->q + h_head_size;
            float* k = s->k + h_head_size;
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for (int i = 0; i < head_size; i+=2) {
                int j = i / 2;
                float q0 = q[i];
                float q1 = q[i+1];
                float k0 = k[i];
                float k1 = k[i+1];
                float fcr = freq_cis_real_row[j];
                float fci = freq_cis_imag_row[j];
                q[i]   = q0 * fcr - q1 * fci;
                q[i+1] = q0 * fci + q1 * fcr;
                k[i]   = k0 * fcr - k1 * fci;
                k[i+1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * dim;
        float* value_cache_row = s->value_cache + loff + pos * dim;
        memcpy(key_cache_row, s->k, dim*sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, dim*sizeof(*value_cache_row));
        
        // multihead attention. iterate over all heads
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            // common expression hoisting
            int h_head_size = h * head_size;

            // get the query vector for this head
            float* q = s->q + h_head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * dim + h_head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score *= inv_sqrt_head_size;
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);
            
            // weighted sum of the values, store back into xb
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    val += att[t] * s->value_cache[loff + t * dim + h_head_size + i]; // note bad locality
                }
                s->xb[h_head_size + i] = val;
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
#ifndef LLAMAC_AVX2
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
#else
        // used fused / loop-jamming version when we have AVX2
        matmul2(s->hb, s->hb2, s->xb, w->w1 + l*dim*hidden_dim, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
#endif

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
        }
        
        // elementwise multiply with w3(x)
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }
    
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
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
    // Get the current time with nanosecond precision
    if (clock_gettime(CLOCK_REALTIME, &time) == 0) {
        return time.tv_sec * 1000 + time.tv_nsec / 1000000;
    } else {
        perror("clock_gettime");
        return -1; // Return -1 to indicate an error
    }
}
int main(int argc, char *argv[]) {

    // poor man's C argparse
    char *checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [temperature] [steps]\n", argv[0]);
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

    // seed rng with time. if you want deterministic behavior use temperature 0.0
    srand((unsigned int)time(NULL)); 
    
    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    int fd = 0;
    float* data = NULL;
    long file_size;
    {
        FILE *file = fopen(checkpoint, "rb");
        if (!file) {
            printf("Unable to open the checkpoint file %s!\n", checkpoint);
            return 1;
        }
        // read in the config header
        if(fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        file_size = ftell(file); // get the file size, in bytes
        fclose(file);
        // memory map the Transformer weights into the data pointer
        fd = open(checkpoint, O_RDONLY); // open in read only mode
        if (fd == -1) { printf("open failed!\n"); return 1; }
        data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

        CHECK_ALIGNMENT(data);

        if (data == MAP_FAILED) { printf("mmap failed!\n"); return 1; }
        float* weights_ptr = data + sizeof(Config)/sizeof(float);
        checkpoint_init_weights(&weights, &config, weights_ptr, shared_weights);
    }
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }

    // read in the tokenizer.bin file
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    {
        FILE *file = fopen("tokenizer.bin", "rb");
        if (!file) {
            printf("Unable to open the tokenizer file tokenizer.bin! Run "
            "python tokenizer.py to convert tokenizer.model -> tokenizer.bin\n");
            return 1;
        }
        int len;
        for (int i = 0; i < config.vocab_size; i++) {
            if(fread(&len, sizeof(int), 1, file) != 1) { return 1; }
            vocab[i] = (char *)malloc(len + 1);
            if(fread(vocab[i], len, 1, file) != 1) { return 1; }
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
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        transformer(token, pos, &config, &state, &weights);

        // sample the next token
        if(temperature == 0.0f) {
            // greedy argmax sampling
            next = argmax(state.logits, config.vocab_size);
        } else {
            // apply the temperature to the logits
            for (int q=0; q<config.vocab_size; q++) { state.logits[q] /= temperature; }
            // apply softmax to the logits to get the probabilities for next token
            softmax(state.logits, config.vocab_size);
            // we now want to sample from this distribution to get the next token
            next = sample(state.logits, config.vocab_size);
        }
        printf("%s", vocab[next]);
        fflush(stdout);

        // advance forward
        token = next;
        pos++;
    }

    // report achieved tok/s
    long end = time_in_ms();
    printf("\nachieved tok/s: %f\n", steps / (double)(end-start)*1000);

    // memory and file handles cleanup
    free_run_state(&state);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    if (data != MAP_FAILED) munmap(data, file_size);
    if (fd != -1) close(fd);
    return 0;
}
