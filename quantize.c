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
    // token embedding table
    int8_t* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    int8_t* rms_att_weight; // (layer, dim) rmsnorm weights
    int8_t* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    int8_t* wq; // (layer, dim, dim)
    int8_t* wk; // (layer, dim, dim)
    int8_t* wv; // (layer, dim, dim)
    int8_t* wo; // (layer, dim, dim)
    // weights for ffn
    int8_t* w1; // (layer, hidden_dim, dim)
    int8_t* w2; // (layer, dim, hidden_dim)
    int8_t* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    int8_t* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    int8_t* freq_cis_real; // (seq_len, dim/2)
    int8_t* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    int8_t* wcls;
} TransformerWeightsInt8;

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

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



void calc_quant_ints(float * ptr, int size, int8_t * out_ptr, float max, int debug){

    int8_t x_quant;
    for (int i = 0; i < size; i++){
        x_quant = round(127/max * ptr[i]);
        out_ptr[i] = x_quant;
    }
};

float get_max_vals(float *ptr, int size){
    float max = -INFINITY;

    for (int i = 0; i < size; i++){
        if (ptr[i] > max) max = ptr[i];
    }
    return max; 
 
};

int convert_q8(TransformerWeights *w, Config *p, int glob_data_size, int elements){

    float max_vals[13];
    int8_t *quant_weights = calloc(elements, sizeof(int8_t));

    FILE* file = fopen("data.bin", "wb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    max_vals[0] = get_max_vals(w->token_embedding_table, p->vocab_size * p->dim);
    max_vals[1] = get_max_vals(w->rms_att_weight, p->n_layers * p->dim);
    max_vals[2] = get_max_vals(w->wq, p->n_layers * p->dim * p->dim);
    max_vals[3] = get_max_vals(w->wk, p->n_layers * p->dim * p->dim);
    max_vals[4] = get_max_vals(w->wv, p->n_layers * p->dim * p->dim);
    max_vals[5] = get_max_vals(w->wo, p->n_layers * p->dim * p->dim);

    max_vals[6] = get_max_vals(w->rms_ffn_weight, p->n_layers * p->dim);
    
    max_vals[7] = get_max_vals(w->w1, p->n_layers * p->dim * p->hidden_dim);
    max_vals[8] = get_max_vals(w->w2, p->n_layers * p->hidden_dim * p->dim);
    max_vals[9] = get_max_vals(w->w3, p->n_layers * p->dim * p->hidden_dim);
    
    max_vals[10] = get_max_vals(w->rms_final_weight, p->dim);

    int head_size = p->dim / p->n_heads;
    max_vals[11] = get_max_vals(w->freq_cis_real, p->seq_len * head_size / 2);
    max_vals[12] = get_max_vals(w->freq_cis_imag, p->seq_len * head_size / 2);

    //for (int i = 0; i < 13; i++) printf("max[%d] = %f\n",i, max_vals[i]);
    // write headers
    fwrite(&p->dim, sizeof(int), 1, file);
    fwrite(&p->hidden_dim, sizeof(int), 1, file);
    fwrite(&p->n_layers, sizeof(int), 1, file);
    fwrite(&p->n_heads, sizeof(int), 1, file);
    fwrite(&p->n_kv_heads, sizeof(int), 1, file);
    fwrite(&p->vocab_size, sizeof(int), 1, file);
    fwrite(&p->seq_len, sizeof(int), 1, file);

    // // write max values
    fwrite(max_vals, sizeof(float), 13, file);

    // // write quantized weights
    int8_t * out_ptr;
    out_ptr = quant_weights;
    calc_quant_ints(w->token_embedding_table, p->vocab_size * p->dim, out_ptr, max_vals[0], 0);
    out_ptr += p->vocab_size * p->dim;
    calc_quant_ints(w->rms_att_weight, p->n_layers * p->dim, out_ptr,  max_vals[1],0);
    out_ptr += p->n_layers * p->dim;
    calc_quant_ints(w->wq, p->n_layers * p->dim * p->dim, out_ptr,  max_vals[2],0);
    out_ptr += p->n_layers * p->dim * p->dim;
    calc_quant_ints(w->wk, p->n_layers * p->dim * p->dim, out_ptr,  max_vals[3],0);
    out_ptr += p->n_layers * p->dim * p->dim;
    calc_quant_ints(w->wv, p->n_layers * p->dim * p->dim, out_ptr,  max_vals[4],0);
    out_ptr += p->n_layers * p->dim * p->dim; 
    calc_quant_ints(w->wo, p->n_layers * p->dim * p->dim, out_ptr,  max_vals[5],0);
    out_ptr += p->n_layers * p->dim * p->dim;
    
    calc_quant_ints(w->rms_ffn_weight, p->n_layers * p->dim, out_ptr, max_vals[6], 0);
    out_ptr += p->n_layers * p->dim;

    calc_quant_ints(w->w1, p->n_layers * p->dim * p->hidden_dim, out_ptr,  max_vals[7],0);
    out_ptr += p->n_layers * p->dim * p->hidden_dim;
    calc_quant_ints(w->w2, p->n_layers * p->hidden_dim * p->dim, out_ptr,  max_vals[8],0);
    out_ptr += p->n_layers * p->hidden_dim * p->dim;
    calc_quant_ints(w->w3, p->n_layers * p->dim * p->hidden_dim, out_ptr,  max_vals[9],0);
    out_ptr += p->n_layers * p->dim * p->hidden_dim;

    calc_quant_ints(w->rms_final_weight, p->dim, out_ptr,  max_vals[10],0);
    out_ptr += p->dim; 

    calc_quant_ints(w->freq_cis_real, p->seq_len * head_size / 2, out_ptr,  max_vals[11],0);
    out_ptr += p->seq_len * head_size / 2;

    calc_quant_ints(w->freq_cis_imag, p->seq_len * head_size / 2, out_ptr,  max_vals[12],1);
    // out_ptr += p->seq_len * head_size / 2;

    fwrite(quant_weights, sizeof(int8_t), elements, file);

    fclose(file);
    return 0;
};

int calculate_num_elements(Config *p){
    int head_size = p->dim / p->n_heads;
    return p->vocab_size * p->dim + 
           p->n_layers * p->dim +
           4 * p->n_layers * p->dim * p->dim +
           p->n_layers * p->dim +
           3 * p->n_layers * p->dim * p->hidden_dim + 
           p->dim +
           p->seq_len * head_size;

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
        //print_sample_weights(&weights);

        int elements = calculate_num_elements(&config);
        //printf("total elements = %d\n", elements);
        int ret = convert_q8(&weights, &config, file_size/sizeof(float), elements);
        if (ret == 0) printf("model converted and saved\n");
    }
    
    // memory and file handles cleanup
    if (data != MAP_FAILED) munmap(data, file_size);
    if (fd != -1) close(fd);
    return 0;
}
