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
#include <fcntl.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

// Configuration for the Transformer model
typedef struct {
    int dim;           // Transformer dimension
    int hidden_dim;    // For ffn layers
    int n_layers;      // Number of layers
    int n_heads;       // Number of query heads
    int n_kv_heads;    // Number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;    // Vocabulary size, usually 256 (byte-level)
    int seq_len;       // Max sequence length
} Config;

// Weights for the Transformer model
typedef struct {
    float* token_embedding_table; // Token embedding table (vocab_size, dim)
    float* rms_att_weight;        // Weights for rmsnorms for attention (layer, dim)
    float* rms_ffn_weight;        // Weights for rmsnorms for feed-forward network (layer, dim)
    float* wq;                   // Weights for matmul of query (layer, dim, dim)
    float* wk;                   // Weights for matmul of key (layer, dim, dim)
    float* wv;                   // Weights for matmul of value (layer, dim, dim)
    float* wo;                   // Weights for matmul output (layer, dim, dim)
    float* w1;                   // Weights for ffn first layer (layer, hidden_dim, dim)
    float* w2;                   // Weights for ffn second layer (layer, dim, hidden_dim)
    float* w3;                   // Weights for ffn third layer (layer, hidden_dim, dim)
    float* rms_final_weight;      // Final rmsnorm weights (dim,)
    float* freq_cis_real;         // Freq_cis for RoPE relatively positional embeddings (seq_len, dim/2)
    float* freq_cis_imag;         // Freq_cis for RoPE relatively positional embeddings (seq_len, dim/2)
    float* wcls;                  // (Optional) Classifier weights for the logits, on the last layer
} TransformerWeights;

// State of the current run (activation values and other intermediate variables)
typedef struct {
    float *x;           // Activation at current time stamp (dim,)
    float *xb;          // Activation at current time stamp inside a residual branch (dim,)
    float *xb2;         // An additional buffer just for convenience (dim,)
    float *hb;          // Buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;         // Buffer for hidden dimension in the ffn (hidden_dim,)
    float *q;           // Query (dim,)
    float *k;           // Key (dim,)
    float *v;           // Value (dim,)
    float *att;         // Buffer for scores/attention values (n_heads, seq_len)
    float *logits;      // Output logits
    float* key_cache;   // Key cache for kv cache (layer, seq_len, dim)
    float* value_cache; // Value cache for kv cache (layer, seq_len, dim)
} RunState;

// Function to allocate memory for RunState struct
void malloc_run_state(RunState* s, Config* p) {
    // We use calloc instead of malloc to initialize all memory to zero.
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(p->dim, sizeof(float));
    s->v = calloc(p->dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    // Ensure all mallocs went fine, if not, exit the program.
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        printf("malloc failed!\n");
        exit(1);
    }
}

// Function to free the memory allocated for RunState struct
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
// Initialization: Read from checkpoint

// Function to initialize TransformerWeights from a checkpoint file
void checkpoint_init_weights(TransformerWeights *w, Config* p, float* f, int shared_weights) {
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
// Neural net blocks

// Function to accumulate values of two arrays element-wise
void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

// Function to perform root mean square normalization
void rmsnorm(float* o, float* x, float* weight, int size) {
    // Calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // Normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

// Function to apply softmax to an array of values
void softmax(float* x, int size) {
    // Find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // Exponentiate and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // Normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Function to perform matrix multiplication (matmul)
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // By far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// Function to process each transformer block
void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {

    // A few convenience variables
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // Copy the token embedding into x
    float* content_row = &(w->token_embedding_table[token * dim]);
    memcpy(x, content_row, dim * sizeof(*x));

    // Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // Forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // Attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l * dim * dim, dim, dim);

        // Apply RoPE rotation to the q and k vectors for each head
        for (int h = 0; h < p->n_heads; h++) {
            // Get the q and k vectors for this head
            float* q = s->q + h * head_size;
            float* k = s->k + h * head_size;
            // Rotate q and k by the freq_cis_real and freq_cis_imag
            for (int i = 0; i < head_size; i += 2) {
                float q0 = q[i];
                float q1 = q[i + 1];
                float k0 = k[i];
                float k1 = k[i + 1];
                float fcr = freq_cis_real_row[i / 2];
                float fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // Save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * dim;
        float* value_cache_row = s->value_cache + loff + pos * dim;
        memcpy(key_cache_row, s->k, dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, dim * sizeof(*value_cache_row));

        // Multihead attention. Iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // Get the query vector for this head
            float* q = s->q + h * head_size;
            // Attention scores for this head
            float* att = s->att + h * p->seq_len;
            // Iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // Get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * dim + h * head_size;
                // Calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // Save the score to the attention buffer
                att[t] = score;
            }

            // Softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // Weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // Get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * dim + h * head_size;
                // Get the attention weight for this timestep
                float a = att[t];
                // Accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // Final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // Residual connection back into x
        accum(x, s->xb2, dim);

        // FFN rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // First calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // F.silu; silu(x) = x * σ(x), where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
        }

        // Element-wise multiply with w3(x)
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        // Final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // Residual connection
        accum(x, s->xb, dim);
    }

    // Final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // Classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

// Function to sample the next token from a probability distribution
int sample(float* probabilities, int n) {
    // Sample index from probabilities, they must sum to 1
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // In case of rounding errors
}

// Function to find the index of the maximum value in an array
int argmax(float* v, int n) {
    // Return argmax of v in elements 0..n
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
// Main function

// Function to get the current time in milliseconds
long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int main(int argc, char *argv[]) {

    // Poor man's C argparse
    char *checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // Max number of steps to run for, 0: use seq_len
    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [temperature] [steps]\n", argv[0]);
        return 1;
    }
    if (argc >= 2) {
        checkpoint = argv[1];
    }
    if (argc >= 3) {
        // Optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
        temperature = atof(argv[2]);
    }
    if (argc >= 4) {
        steps = atoi(argv[3]);
    }

    // Seed RNG with time. If you want deterministic behavior, use temperature 0.0
    srand((unsigned int)time(NULL));

    // Read in the model.bin file
    Config config;
    TransformerWeights weights;
    int fd = 0;         // File descriptor for memory mapping
    float* data = NULL; // Memory mapped data pointer
    long file_size;     // Size of the file
    // Open file for reading in binary mode
    fd = open(checkpoint, O_RDONLY | O_BINARY);
    if (fd < 0) {
        printf("Failed to open checkpoint file '%s'\n", checkpoint);
        return 1;
    }
    // Get the file size
    file_size = lseek(fd, 0, SEEK_END);
    // Memory map the file
    data = (float*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (!data) {
        printf("Failed to map checkpoint file '%s'\n", checkpoint);
        close(fd);
        return 1;
    }
    // Extract config from the beginning of the file
    memcpy(&config, data, sizeof(config));
    // Set the pointer to point after the config
    data += sizeof(config);

    // Set the pointer to point to the weights (after the config)
    checkpoint_init_weights(&weights, &config, data, 0);

    // Print model configuration
    printf("Configuration:\n");
    printf("dim: %d\n", config.dim);
    printf("hidden_dim: %d\n", config.hidden_dim);
    printf("n_layers: %d\n", config.n_layers);
    printf("n_heads: %d\n", config.n_heads);
    printf("vocab_size: %d\n", config.vocab_size);
    printf("seq_len: %d\n", config.seq_len);

    // Allocate memory for the RunState struct
    RunState state;
    malloc_run_state(&state, &config);

    // Initial token to start generation
    int token = 0; // start of sequence token
    // Number of steps, 0 means use config.seq_len
    int T = steps > 0 ? steps : config.seq_len;
    printf("Running for %d steps\n", T);

    // Start the generation loop
    long start_time = time_in_ms();
    for (int t = 0; t < T; t++) {
        // Calculate position, which will be used for relative positional embedding (RoPE)
        int pos = t;
        // Run transformer for one step
        transformer(token, pos, &config, &state, &weights);
        // Apply temperature and sample the next token
        if (temperature == 0.0f) {
            // Deterministic argmax sampling
            token = argmax(state.logits, config.vocab_size);
        } else {
            // Probabilistic sampling
            for (int i = 0; i < config.vocab_size; i++) {
                state.logits[i] /= temperature;
            }
            softmax(state.logits, config.vocab_size);
            token = sample(state.logits, config.vocab_size);
        }
        // Print the sampled token as a byte
        printf("%c", (char)token);
        // Flush stdout to see the intermediate result in case of large T
        fflush(stdout);
    }
    long end_time = time_in_ms();
    printf("\nGeneration complete in %ld ms\n", end_time - start_time);

    // Clean up
    free_run_state(&state);
    munmap((void *)weights.token_embedding_table, file_size);
    close(fd);

    return 0;
}
