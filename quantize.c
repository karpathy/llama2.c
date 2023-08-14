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
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_sizes
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
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

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

void checkpoint_init_weights(TransformerWeights *w, Config* p, float* f, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    float* ptr = f;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->wq = ptr;
    ptr += p->n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += p->n_layers * (p->n_heads * head_size) * p->dim;
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
    ptr += p->seq_len * head_size / 2;
    w->freq_cis_imag = ptr;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void get_minmax(float *ptr, int size, float* pmin, float* pmax){
    float min = INFINITY;
    float max = -INFINITY;

    for (int i = 0; i < size; i++){
        if (ptr[i] < min) min = ptr[i];
        if (ptr[i] > max) max = ptr[i];
    }

    *pmin = min;
    *pmax = max;
}

void quantize_weights(FILE* file, float *weights, int n_layers, int layer_size, char *name) {

    puts("------------------------");
    printf("%s layer_size=%d\n", name, layer_size);

    // for each layer
    for (int l = 0; l < n_layers; l++) {
      // get the min and max values for this layer
      float min;
      float max;
      get_minmax(weights, layer_size, &min, &max);
      // compute the scale factor
      float scale = (max - min) / 255;
      printf("l=%d min=%f max=%f scale=%f\n", l, min, max, scale);
      // save min value and scale factor to file
      fwrite(&min, sizeof(float), 1, file);
      fwrite(&scale, sizeof(float), 1, file);
      printf("[debug] min, scale written success for layer %d\n", l);
      // quantize the weights from this layer and save to file
      uint8_t qweight;
      for (int i = 0; i < layer_size; i++){
          qweight = round((weights[i] - min) / (max - min) * 255);
          fwrite(&qweight, sizeof(uint8_t), 1, file);
      }
      // advance to the weights of the next layer
      weights += layer_size;  // * sizeof(float);
    }

}

void write_weights(FILE* file, float *weights, int n_layers, int layer_size) {
    fwrite(weights, sizeof(float), n_layers * layer_size, file);
}

int convert_weights_q8(TransformerWeights *w, Config *p){

    FILE* file = fopen("data.bin", "wb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // write headers
    fwrite(&p->dim, sizeof(int), 1, file);
    fwrite(&p->hidden_dim, sizeof(int), 1, file);
    fwrite(&p->n_layers, sizeof(int), 1, file);
    fwrite(&p->n_heads, sizeof(int), 1, file);
    fwrite(&p->n_kv_heads, sizeof(int), 1, file);
    fwrite(&p->vocab_size, sizeof(int), 1, file);
    fwrite(&p->seq_len, sizeof(int), 1, file);

    // write quantized weights
    int head_size = p->dim / p->n_heads;

    quantize_weights(file, w->token_embedding_table, 1, p->vocab_size * p->dim, "token_embedding_table");

    quantize_weights(file, w->rms_att_weight, p->n_layers, p->dim, "rms_att_weight");

    quantize_weights(file, w->wq, p->n_layers, p->dim * (p->n_heads * head_size), "wq");
    quantize_weights(file, w->wk, p->n_layers, p->dim * (p->n_kv_heads * head_size), "wk");
    quantize_weights(file, w->wv, p->n_layers, p->dim * (p->n_kv_heads * head_size), "wv");
    quantize_weights(file, w->wo, p->n_layers, (p->n_heads * head_size) * p->dim, "wo");

    quantize_weights(file, w->rms_ffn_weight, p->n_layers, p->dim, "rms_ffn_weight");
    
    quantize_weights(file, w->w1, p->n_layers, p->dim * p->hidden_dim, "w1");
    quantize_weights(file, w->w2, p->n_layers, p->hidden_dim * p->dim, "w2");
    quantize_weights(file, w->w3, p->n_layers, p->dim * p->hidden_dim, "w3");

    quantize_weights(file, w->rms_final_weight, 1, p->dim, "rms_final_weight");

    write_weights(file, w->freq_cis_real, 1, p->seq_len * head_size / 2);
    write_weights(file, w->freq_cis_imag, 1, p->seq_len * head_size / 2);

    //quantize_weights(file, w->token_embedding_table, 1, p->vocab_size * p->dim, "wcls");

    fclose(file);
    return 0;
}

int main(int argc, char *argv[]) {

    // poor man's C argparse
    char *checkpoint = NULL;  // e.g. out/model.bin

    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file>\n", argv[0]);
        return 1;
    }
    if (argc >= 2) {
        checkpoint = argv[1];
    }
    
    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    int fd = 0;         // file descriptor for memory mapping
    float* data = NULL; // memory mapped data pointer
    long file_size;     // size of the checkpoint file in bytes
    {
        FILE *file = fopen(checkpoint, "rb");
        if (!file) { printf("Couldn't open file %s\n", checkpoint); return 1; }

        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        printf("vocab size = %d  shared_weights=%d\n", config.vocab_size, shared_weights);

        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        file_size = ftell(file); // get the file size, in bytes
        fclose(file);
        printf("Model file size = %ldMB\n", file_size/1024/1024);

        // // memory map the Transformer weights into the data pointer
        fd = open(checkpoint, O_RDONLY); // open in read only mode
        if (fd == -1) { printf("open failed!\n"); return 1; }
        data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { printf("mmap failed!\n"); return 1; }
        
        // fast-forward to weight data, skipping metadata
        float* weights_ptr = data + sizeof(Config)/sizeof(float);
        
        checkpoint_init_weights(&weights, &config, weights_ptr, shared_weights);

        int ret = convert_weights_q8(&weights, &config);
        if (ret == 0) printf("model converted and saved\n");
    }
    
    // memory and file handles cleanup
    if (data != MAP_FAILED) munmap(data, file_size);
    if (fd != -1) close(fd);
    return 0;
}
