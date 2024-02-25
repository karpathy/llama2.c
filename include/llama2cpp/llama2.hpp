#ifndef LLAMA2CPP_LLAMA2_HPP
#define LLAMA2CPP_LLAMA2_HPP
#include <string>
#include <llama2cpp/transformer.hpp>
#include <llama2cpp/tokenizer.hpp>
#include <llama2cpp/sampler.hpp>

namespace llama2cpp
{

    long time_in_ms()
    {
        // return time in milliseconds, for benchmarking the model speed
        struct timespec time;
        clock_gettime(CLOCK_REALTIME, &time);
        return time.tv_sec * 1000 + time.tv_nsec / 1000000;
    }
    
    void safe_printf(char *piece)
    {
        // piece might be a raw byte token, and we only want to print printable chars or whitespace
        // because some of the other bytes can be various control codes, backspace, etc.
        if (piece == NULL)
        {
            return;
        }
        if (piece[0] == '\0')
        {
            return;
        }
        if (piece[1] == '\0')
        {
            unsigned char byte_val = piece[0];
            if (!(isprint(byte_val) || isspace(byte_val)))
            {
                return; // bad byte, don't print it
            }
        }
        printf("%s", piece);
    }

    void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps)
    {
        char *empty_prompt = "";
        if (prompt == NULL)
        {
            prompt = empty_prompt;
        }

        // encode the (string) prompt into tokens sequence
        int num_prompt_tokens = 0;
        int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
        encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
        if (num_prompt_tokens < 1)
        {
            fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
            exit(EXIT_FAILURE);
        }

        // start the main loop
        long start = 0;               // used to time our code, only initialized after first iteration
        int next;                     // will store the next token in the sequence
        int token = prompt_tokens[0]; // kick off with the first token in the prompt
        int pos = 0;                  // position in the sequence
        while (pos < steps)
        {

            // forward the transformer to get logits for the next token
            float *logits = forward(transformer, token, pos);

            // advance the state machine
            if (pos < num_prompt_tokens - 1)
            {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens[pos + 1];
            }
            else
            {
                // otherwise sample the next token from the logits
                next = sample(sampler, logits);
            }
            pos++;

            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            if (next == 1)
            {
                break;
            }

            // print the token as string, decode it with the Tokenizer object
            char *piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
            token = next;

            // init the timer here because the first iteration can be slower
            if (start == 0)
            {
                start = time_in_ms();
            }
        }
        printf("\n");

        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if (pos > 1)
        {
            long end = time_in_ms();
            fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
        }

        free(prompt_tokens);
    }

    void read_stdin(const char *guide, char *buffer, size_t bufsize)
    {
        // read a line from stdin, up to but not including \n
        printf("%s", guide);
        if (fgets(buffer, bufsize, stdin) != NULL)
        {
            size_t len = strlen(buffer);
            if (len > 0 && buffer[len - 1] == '\n')
            {
                buffer[len - 1] = '\0'; // strip newline
            }
        }
    }

    // ----------------------------------------------------------------------------
    // chat loop
    // I manually inspected the tokens for a few chat conversations compared to
    // python reference and that seemed ok, but this was not thoroughly tested and
    // is not safely implemented, it's more a proof of concept atm.

    void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
              char *cli_user_prompt, char *cli_system_prompt, int steps)
    {

        // buffers for reading the system prompt and user prompt from stdin
        // you'll notice they are soomewhat haphazardly and unsafely set atm
        char system_prompt[512];
        char user_prompt[512];
        char rendered_prompt[1152];
        int num_prompt_tokens = 0;
        int *prompt_tokens = (int *)malloc(1152 * sizeof(int));
        int user_idx;

        // start the main loop
        int8_t user_turn = 1; // user starts
        int next;             // will store the next token in the sequence
        int token;            // stores the current token to feed into the transformer
        int prev_token;
        int pos = 0; // position in the sequence
        while (pos < steps)
        {

            // when it is the user's turn to contribute tokens to the dialog...
            if (user_turn)
            {
                // get the (optional) system prompt at position 0
                if (pos == 0)
                {
                    // at position 0, the user can also contribute a system prompt
                    if (cli_system_prompt == NULL)
                    {
                        // system prompt was not passed in, attempt to get it from stdin
                        read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                    }
                    else
                    {
                        // system prompt was passed in, use it
                        strcpy(system_prompt, cli_system_prompt);
                    }
                }
                // get the user prompt
                if (pos == 0 && cli_user_prompt != NULL)
                {
                    // user prompt for position 0 was passed in, use it
                    strcpy(user_prompt, cli_user_prompt);
                }
                else
                {
                    // otherwise get user prompt from stdin
                    read_stdin("User: ", user_prompt, sizeof(user_prompt));
                }
                // render user/system prompts into the Llama 2 Chat schema
                if (pos == 0 && system_prompt[0] != '\0')
                {
                    char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                    sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
                }
                else
                {
                    char user_template[] = "[INST] %s [/INST]";
                    sprintf(rendered_prompt, user_template, user_prompt);
                }
                // encode the rendered prompt into tokens
                encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
                user_idx = 0; // reset the user index
                user_turn = 0;
                printf("Assistant: ");
            }

            // determine the token to pass into the transformer next
            if (user_idx < num_prompt_tokens)
            {
                // if we are still processing the input prompt, force the next prompt token
                token = prompt_tokens[user_idx++];
            }
            else
            {
                // otherwise use the next token sampled from previous turn
                token = next;
            }
            // EOS (=2) token ends the Assistant turn
            if (token == 2)
            {
                user_turn = 1;
            }

            // forward the transformer to get logits for the next token
            float *logits = forward(transformer, token, pos);
            next = sample(sampler, logits);
            pos++;

            if (user_idx >= num_prompt_tokens && next != 2)
            {
                // the Assistant is responding, so print its output
                char *piece = decode(tokenizer, token, next);
                safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
                fflush(stdout);
            }
            if (next == 2)
            {
                printf("\n");
            }
        }
        printf("\n");
        free(prompt_tokens);
    }

}
#endif