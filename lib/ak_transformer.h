#ifndef ak_transformer_H
#define ak_transformer_H

// ----------------------------------------------------------------------------
// Transformer model
struct ak_transformer_s;
typedef struct ak_transformer_s ak_transformer_t;

ak_transformer_t* ak_transformer_init(char* checkpoint_path);
void ak_transformer_destroy(ak_transformer_t* t);

int ak_transformer_seq_len(ak_transformer_t* t);
int ak_transformer_vocab_size(ak_transformer_t* t);

// runs the forward pass on the transformer
float* ak_transformer_forward(ak_transformer_t* transformer, int token, int pos);

#include "ak_transformer.c"

#endif
