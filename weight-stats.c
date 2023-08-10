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

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void print_sample_weights(TransformerWeights *w){
    printf("----- Quick print of first of the weight vales of all the variables\n");
    printf("%f\n", w->token_embedding_table[0]);
    printf("%f\n", w->rms_att_weight[0]);
    printf("%f\n", w->rms_ffn_weight[0]);

    printf("%f\n", w->wq[0]);
    printf("%f\n", w->wk[0]);
    printf("%f\n", w->wv[0]);
    printf("%f\n", w->wo[0]);
    printf("%f\n", w->w1[0]);
    printf("%f\n", w->w2[0]);
    printf("%f\n", w->w3[0]);
    printf("%f\n", w->rms_att_weight[0]);
}

union f2u {
    float f;
    uint32_t u;
};

const size_t HISTOGRAM_BUCKETS = 150;


void calcHistogram(const float* data, int dataSize, float minRange, float maxRange, int* histogram){
//    maxRange += 0.01;
//    minRange -= 0.01;
   float binWidth = (maxRange - minRange) / HISTOGRAM_BUCKETS;
   printf("DEBUG:: binWidth=%f\n", binWidth);
   printf("DEBUG:: dataSize=%d\n", dataSize);
    for (int i = 0; i < dataSize; ++i) {
        if (data[i] >= minRange && data[i] <= maxRange) {
            int binIndex = (int)((data[i] - minRange) / binWidth);
            if (binIndex >= 0 && binIndex < HISTOGRAM_BUCKETS) {
                histogram[binIndex]++;
            }
        }
    }
};

void plotHistogram(int* histogram, float min, float max, char* name){
    char png_file_name[150];
    char set_output[150];
    sprintf(png_file_name, "%s%s%s", "pngs/", name, ".png");
    sprintf(set_output, "%s%s%s", "set output '", png_file_name, "\n");
    //printf(">>>>>>>>>> %s", png_file_name);

    char title[150];
    //sprintf(title, "%s %s %s %.2f %s %.2f %s","set title", png_file_name,"Histogram (Min: ", min, ", Max: ", max, ")'\n");
    sprintf(title, "%s%s%s%.4f%s%.4f%s","set title '[ ", name," ] Histogram (Min: ", min, ", Max: ", max, ")'\n");

    FILE* gp = popen("gnuplot", "w");
    fprintf(gp, "set terminal png\n");                 // Set terminal to PNG format
    //fprintf(gp, "set output 'histogram_plot.png'\n"); // Specify output file name
    fprintf(gp, set_output); // Specify output file name
    fprintf(gp, "set style data histogram\n");
    fprintf(gp, "set xlabel 'Bins'\n");
    //fprintf(gp, "set title 'Histogram (Min: %.2f, Max: %.2f)'\n", min, max);
    fprintf(gp, title);
    fprintf(gp, "plot '-' using 2:xticlabels(1) title 'Frequency'\n");
    for (int i = 0; i < HISTOGRAM_BUCKETS; ++i) {
        fprintf(gp, "%d %d\n", i, histogram[i]);
    }
    fprintf(gp, "e\n");
    pclose(gp); 

};

void calc_stats(float * ptr, int size, char* name,  int glob_data_size){
   union f2u max, min;
    max.f = -INFINITY;
    min.f = INFINITY;

    for (int i = 0; i < size; i++){
        if (ptr[i] < min.f) min.f = ptr[i];
        if (ptr[i] > max.f) max.f = ptr[i];
    }

    printf("-----------%s\n", name);
    printf("min = 0x%0x [%f], max = 0x%0x [%f] [%d/%d(%.2f%%)]\n", min.u, min.f, max.u, max.f, size, glob_data_size, ((float)(size)/glob_data_size)*100.0);
    printf("-----------\n");
    int histogram[HISTOGRAM_BUCKETS];
    for (int i = 0; i < HISTOGRAM_BUCKETS; ++i) {
        histogram[i] = 0;
    }
    calcHistogram(ptr, size, min.f, max.f, histogram);
    plotHistogram(histogram, min.f, max.f, name);

};

void get_stats(TransformerWeights *w, Config *p, int glob_data_size){

    calc_stats(w->token_embedding_table, p->vocab_size * p->dim, "tet", glob_data_size);
    calc_stats(w->rms_att_weight, p->n_layers * p->dim, "rms-att", glob_data_size);
    calc_stats(w->wq, p->n_layers * p->dim * p->dim, "wq", glob_data_size);
    calc_stats(w->wk, p->n_layers * p->dim * p->dim, "qk", glob_data_size);
    calc_stats(w->wv, p->n_layers * p->dim * p->dim, "qv", glob_data_size);
    calc_stats(w->wo, p->n_layers * p->dim * p->dim, "wo", glob_data_size);

    calc_stats(w->rms_ffn_weight, p->n_layers * p->dim, "ffn" , glob_data_size);
    
    calc_stats(w->w1, p->n_layers * p->dim * p->hidden_dim, "w1", glob_data_size);
    calc_stats(w->w2, p->n_layers * p->hidden_dim * p->dim, "w2", glob_data_size);
    calc_stats(w->w3, p->n_layers * p->dim * p->hidden_dim, "w3", glob_data_size);
    
    calc_stats(w->rms_final_weight, p->dim, "rms-final", glob_data_size);

    int head_size = p->dim / p->n_heads;
    calc_stats(w->freq_cis_real, p->seq_len * head_size / 2, "freqi-real", glob_data_size);
    calc_stats(w->freq_cis_imag, p->seq_len * head_size / 2, "freq-imag", glob_data_size);

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
        // Here we are only opening the file traversing it to the end to calcualte the size of it. ////////////////////////////
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
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // // memory map the Transformer weights into the data pointer
        fd = open(checkpoint, O_RDONLY); // open in read only mode
        if (fd == -1) { printf("open failed!\n"); return 1; }
        data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { printf("mmap failed!\n"); return 1; }
        
        // fast-forward to weight data, skipping metadata
        float* weights_ptr = data + sizeof(Config)/sizeof(float);
        
        checkpoint_init_weights(&weights, &config, weights_ptr, shared_weights);
        //print_sample_weights(&weights);

        get_stats(&weights, &config, file_size/sizeof(float));
//        calc_stats(weights_ptr, file_size/sizeof(float) - sizeof(Config)/sizeof(float), "full");
    }
    
    // memory and file handles cleanup
    if (data != MAP_FAILED) munmap(data, file_size);
    if (fd != -1) close(fd);
    return 0;
}
