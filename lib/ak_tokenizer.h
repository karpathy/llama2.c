#ifndef ak_tokenizer_H
#define ak_tokenizer_H

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

struct ak_tokenizer_s;
typedef struct ak_tokenizer_s ak_tokenizer_t;

ak_tokenizer_t *ak_tokenizer_init(char* tokenizer_path, int vocab_size);
char* ak_tokenizer_decode(ak_tokenizer_t* t, int prev_token, int token);
void ak_tokenizer_encode(ak_tokenizer_t* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
void ak_tokenizer_destroy(ak_tokenizer_t* t);

#include "ak_tokenizer.c"

#endif /* ak_tokenizer_H */
