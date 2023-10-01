#ifndef ak_tokenizer_H
#define ak_tokenizer_H

#include <stdbool.h>

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

struct ak_tokenizer_s;
typedef struct ak_tokenizer_s ak_tokenizer_t;

ak_tokenizer_t *ak_tokenizer_init(char* tokenizer_path, int vocab_size);
void ak_tokenizer_destroy(ak_tokenizer_t* t);

// The prev_token is only needed to to strip leading whitespace following a BOS (beginning of sentence)
// Otherwise, the token is converted to the text version.
// TODO: Maybe this should return const char* ?
char* ak_tokenizer_decode(ak_tokenizer_t* t, int prev_token, int token);

// Encodes text into a series of n_tokens tokens with optional bos (1) or eos (2).  Each token is an int.
/*
    This will read the text one utf-8 character at a time.  If utf-8 sequences are wrong, the token will
    split into individual bytes.

    TODO: I'm assuming in the dictionary of 50k tokens that the first 256 token are equivalent to
        single byte(s).  Check that this is true.

    Once tokens are found, there is another pass which will try and merge shorter tokens into longer tokens
    based upon a score in the tokenizer dictionary (what is stored on disk).
*/
void ak_tokenizer_encode(ak_tokenizer_t* t, char *text, bool bos, bool eos, int *tokens, int *n_tokens);

#include "ak_tokenizer.c"

#endif /* ak_tokenizer_H */
