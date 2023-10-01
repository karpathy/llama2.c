/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
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

#include <stdbool.h>

#include "ak_utils.h"
#include "ak_softmax.h"
#include "ak_rmsnorm.h"
#include "ak_sampler.h"
#include "ak_matmul.h"
#include "ak_tokenizer.h"
#include "ak_transformer.h"
#include "ak_params.h"

// ----------------------------------------------------------------------------
// generation loop

void generate(ak_transformer_t *transformer, ak_tokenizer_t *tokenizer, ak_sampler_t *sampler, char *prompt, int steps) {
    char *empty_prompt = (char *)"";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    ak_tokenizer_encode(tokenizer, prompt, true, false, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = ak_transformer_forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = ak_sampler_sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = ak_tokenizer_decode(tokenizer, token, next);
        ak_utils_safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = ak_utils_time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = ak_utils_time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(ak_transformer_t *transformer, ak_tokenizer_t *tokenizer, ak_sampler_t *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    bool user_turn = true; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    ak_utils_read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                ak_utils_read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                snprintf(rendered_prompt, 1152, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                snprintf(rendered_prompt, 1152, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            ak_tokenizer_encode(tokenizer, rendered_prompt, true, false, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = false;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = true; }

        // forward the transformer to get logits for the next token
        float* logits = ak_transformer_forward(transformer, token, pos);
        next = ak_sampler_sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = ak_tokenizer_decode(tokenizer, token, next);
            ak_utils_safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

int main(int argc, char *argv[]) {
    ak_params_t *params = ak_params_init(argc, argv);

    // build the Transformer via the model .bin file
    ak_transformer_t* transformer = ak_transformer_init(params->checkpoint_path);
    if (params->steps == 0 || params->steps > ak_transformer_seq_len(transformer))
        params->steps = ak_transformer_seq_len(transformer); // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    ak_tokenizer_t* tokenizer = ak_tokenizer_init(params->tokenizer_path, ak_transformer_vocab_size(transformer));

    // build the Sampler
    ak_sampler_t* sampler = ak_sampler_init(ak_transformer_vocab_size(transformer),
                                            params->temperature, params->topp, params->rng_seed);

    // run!
    if (strcmp(params->mode, "generate") == 0) {
        generate(transformer, tokenizer, sampler, params->prompt, params->steps);
    } else if (strcmp(params->mode, "chat") == 0) {
        chat(transformer, tokenizer, sampler, params->prompt, params->system_prompt, params->steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", params->mode);
        ak_params_error_usage();
    }

    // memory and file handles cleanup
    ak_sampler_destroy(sampler);
    ak_tokenizer_destroy(tokenizer);
    ak_transformer_destroy(transformer);
    ak_params_destroy(params);
    return 0;
}
#endif
