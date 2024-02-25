#ifndef LLAMA2CPP_TOKENIZER_HPP
#define LLAMA2CPP_TOKENIZER_HPP
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <memory>

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

namespace llama2cpp
{
    static constexpr const size_t BYTE_STR_SIZE = 512;

    struct TokenIndex
    {
        char *str;
        int id;
    };

    int compare_tokens(const void *a, const void *b)
    {
        return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
    }

    int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
    {
        // efficiently find the perfect match for str in vocab, return its index or -1 if not found
        TokenIndex tok = {.str = str}; // acts as the key to search for
        TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
        return res != NULL ? res->id : -1;
    }

    /**
     * @brief Tokenize prompt
     *
     * @TODO: rewrite as class.
     *
     */
    class Tokenizer
    {
    public:
        using ptr = std::unique_ptr<Tokenizer>;

        Tokenizer(const std::string &tokenizer_path, int vocab_size_)
        {
            // i should have written the vocab_size into the tokenizer file... sigh
            vocab_size = vocab_size_;
            // malloc space to hold the scores and the strings
            vocab = (char **)malloc(vocab_size * sizeof(char *));
            vocab_scores = (float *)malloc(vocab_size * sizeof(float));
            sorted_vocab = NULL; // initialized lazily
            for (int i = 0; i < 256; i++)
            {
                byte_pieces[i * 2] = (unsigned char)i;
                byte_pieces[i * 2 + 1] = '\0';
            }
            // read in the file
            FILE *file = fopen(tokenizer_path.c_str(), "rb");
            if (!file)
            {
                fprintf(stderr, "couldn't load %s\n", tokenizer_path.c_str());
                exit(EXIT_FAILURE);
            }
            if (fread(&max_token_length, sizeof(int), 1, file) != 1)
            {
                fprintf(stderr, "failed read\n");
                exit(EXIT_FAILURE);
            }
            int len;
            for (int i = 0; i < vocab_size; i++)
            {
                if (fread(vocab_scores + i, sizeof(float), 1, file) != 1)
                {
                    fprintf(stderr, "failed read\n");
                    exit(EXIT_FAILURE);
                }
                if (fread(&len, sizeof(int), 1, file) != 1)
                {
                    fprintf(stderr, "failed read\n");
                    exit(EXIT_FAILURE);
                }
                vocab[i] = (char *)malloc(len + 1);
                if (fread(vocab[i], len, 1, file) != 1)
                {
                    fprintf(stderr, "failed read\n");
                    exit(EXIT_FAILURE);
                }
                vocab[i][len] = '\0'; // add the string terminating token
            }
            fclose(file);
        }

        ~Tokenizer()
        {
            for (auto i = 0; i < vocab_size; i++)
            {
                free(vocab[i]);
            }
            free(vocab);
            free(vocab_scores);
            free(sorted_vocab);
        }

        void encode(std::string text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
        {
            // encode the string text (input) into an upper-bound preallocated tokens[] array
            // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)

            if (sorted_vocab == NULL)
            {
                // lazily malloc and sort the vocabulary
                sorted_vocab = (TokenIndex *)malloc(vocab_size * sizeof(TokenIndex));
                for (int i = 0; i < vocab_size; i++)
                {
                    sorted_vocab[i].str = vocab[i];
                    sorted_vocab[i].id = i;
                }
                qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
            }

            // create a temporary buffer that will store merge candidates of always two consecutive tokens
            // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
            char *str_buffer = (char *)malloc((max_token_length * 2 + 1 + 2) * sizeof(char));
            size_t str_len = 0;

            // start at 0 tokens
            *n_tokens = 0;

            // add optional BOS (=1) token, if desired
            if (bos)
                tokens[(*n_tokens)++] = 1;

            // add_dummy_prefix is true by default
            // so prepend a dummy prefix token to the input string, but only if text != ""
            // TODO: pretty sure this isn't correct in the general case but I don't have the
            // energy to read more of the sentencepiece code to figure out what it's doing
            if (text[0] != '\0')
            {
                int dummy_prefix = str_lookup(" ", sorted_vocab, vocab_size);
                tokens[(*n_tokens)++] = dummy_prefix;
            }

            // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
            // Code point ↔ UTF-8 conversion
            // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
            // U+0000	U+007F	    0xxxxxxx
            // U+0080	U+07FF	    110xxxxx	10xxxxxx
            // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
            // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

            // process the raw (UTF-8) byte sequence of the input string
            for (auto c = text.begin(); c != text.end(); ++c)
            {

                // reset buffer if the current byte is ASCII or a leading byte
                // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
                // 0x80 is 10000000
                // in UTF-8, all continuation bytes start with "10" in first two bits
                // so in English this is: "if this byte is not a continuation byte"
                if ((*c & 0xC0) != 0x80)
                {
                    // this byte must be either a leading byte (11...) or an ASCII char (0x...)
                    // => reset our location, as we're starting a new UTF-8 codepoint
                    str_len = 0;
                }

                // append the current byte to the buffer
                str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
                str_buffer[str_len] = '\0';

                // while the next character is a continuation byte, continue appending
                // but if there are too many of them, just stop to avoid overruning str_buffer size.
                if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
                {
                    continue;
                }

                // ok c+1 is not a continuation byte, so we've read in a full codepoint
                int id = str_lookup(str_buffer, sorted_vocab, vocab_size);

                if (id != -1)
                {
                    // we found this codepoint in vocab, add it as a token
                    tokens[(*n_tokens)++] = id;
                }
                else
                {
                    // byte_fallback encoding: just encode each byte as a token
                    // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                    // so the individual bytes only start at index 3
                    for (int i = 0; i < str_len; i++)
                    {
                        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
                    }
                }
                str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
            }

            // merge the best consecutive pair each iteration, according the scores in vocab_scores
            while (1)
            {
                float best_score = -1e10;
                int best_id = -1;
                int best_idx = -1;

                for (int i = 0; i < (*n_tokens - 1); i++)
                {
                    // check if we can merge the pair (tokens[i], tokens[i+1])
                    sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i + 1]]);
                    int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
                    if (id != -1 && vocab_scores[id] > best_score)
                    {
                        // this merge pair exists in vocab! record its score and position
                        best_score = vocab_scores[id];
                        best_id = id;
                        best_idx = i;
                    }
                }

                if (best_idx == -1)
                {
                    break; // we couldn't find any more pairs to merge, so we're done
                }

                // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
                tokens[best_idx] = best_id;
                // delete token at position best_idx+1, shift the entire sequence back 1
                for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
                {
                    tokens[i] = tokens[i + 1];
                }
                (*n_tokens)--; // token length decreased
            }

            // add optional EOS (=2) token, if desired
            if (eos)
                tokens[(*n_tokens)++] = 2;

            free(str_buffer);
        }

        auto decode(int prev_token, int token) -> char *
        {
            char *piece = vocab[token];
            // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
            if (prev_token == 1 && piece[0] == ' ')
            {
                piece++;
            }
            // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
            // parse this and convert and return the actual byte
            unsigned char byte_val;
            if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
            {
                piece = (char *)byte_pieces + byte_val * 2;
            }
            return piece;
        }

    private:
        char **vocab;
        float *vocab_scores;
        TokenIndex *sorted_vocab;
        int vocab_size;
        unsigned int max_token_length;
        unsigned char byte_pieces[BYTE_STR_SIZE]; // stores all single-byte strings
    };

}
#endif