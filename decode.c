#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
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
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

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

int main(int argc, char *argv[]) {
    char *tokenizer_path = "data/tok1024.bin";
    char *new_tok_path = "data/decode1024.tok";
    char strbuf[16];
    int16_t header[2];

    FILE *file = fopen(new_tok_path, "w");
    if (!file) { fprintf(stderr, "couldn't load %s\n", new_tok_path); exit(EXIT_FAILURE); }

    Tokenizer tokenizer;
    int max = 0;

    build_tokenizer(&tokenizer, tokenizer_path, 1024);

    header[0] = 16; // each token's char array is 16 chars long
    header[1] = 1024; // tokens
    if (fwrite(header, sizeof(int16_t), 2, file) != 2) { fprintf(stderr, "failed write header\n"); exit(EXIT_FAILURE); }

    for (int i = 0; i < 1024; i++) {
        char* tok_str = decode(&tokenizer, 0, i);

        // from safe_printf
        // piece might be a raw byte token, and we only want to print printable chars or whitespace
        // because some of the other bytes can be various control codes, backspace, etc.
        if (tok_str == NULL || tok_str[0] == '\0') { 
            // fill with \0
            for (int j = 0; j < 16; j++) strbuf[j] = 0;
        } else if (tok_str[1] == '\0' && !(isprint(tok_str[0]) || isspace(tok_str[0]))) {
            // bad byte, don't print anything
            for (int j = 0; j < 16; j++) strbuf[j] = 0;
        } else {
            snprintf(strbuf, 16, "%s", tok_str);
        }

        if (fwrite(strbuf, sizeof(char), 16, file) != 16) { fprintf(stderr, "failed write content\n"); exit(EXIT_FAILURE); }
        if (max < strlen(tok_str)) max = strlen(tok_str);
    }
    
    fclose(file);

    file = fopen(new_tok_path, "r");
    if (fread(header, sizeof(int16_t), 2, file) != 2) { fprintf(stderr, "failed read header\n"); exit(EXIT_FAILURE); }
    printf("%d %d\n", header[0], header[1]);
    for (int i = 0; i < 1024; i++) {
        if (fread(strbuf, sizeof(char), 16, file) != 16) { fprintf(stderr, "failed read content\n"); exit(EXIT_FAILURE); }
        printf("%d | In file: %s | In tok: ", i, strbuf);
        safe_printf(decode(&tokenizer, 0, i));
        printf("\n");
    }
    fclose(file);
    
    printf("Max len: %d\n", max);
    free_tokenizer(&tokenizer);
    return 0;
}