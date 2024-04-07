#ifndef LLAMA2CPP_LLAMA2_HPP
#define LLAMA2CPP_LLAMA2_HPP
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <llama2cpp/transformer.hpp>
#include <llama2cpp/tokenizer.hpp>
#include <llama2cpp/sampler.hpp>
#include <llama2cpp/tensor.hpp>
#include <llama2cpp/types.hpp>

namespace llama2cpp
{

    long time_in_ms()
    {
        // return time in milliseconds, for benchmarking the model speed
        struct timespec time;
        clock_gettime(CLOCK_REALTIME, &time);
        return time.tv_sec * 1000 + time.tv_nsec / 1000000;
    }

    void safe_printf(const std::string &piece)
    {
        // piece might be a raw byte token, and we only want to print printable chars or whitespace
        // because some of the other bytes can be various control codes, backspace, etc.
        if (piece.empty())
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
        // printf("%s", piece);
        std::cout << piece;
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

    /**
     * @brief LLama2 app config.
     *
     */
    struct Llama2Config
    {
        // default parameters
        std::string checkpoint_path = ""; // e.g. out/model.bin
        std::string tokenizer_path = "../tokenizer.bin";
        float32_t temperature = 1.0f;    // 0.0 = greedy deterministic. 1.0 = original. don't set higher
        float32_t topp = 0.9f;           // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
        int32_t steps = 256;             // number of steps to run for
        unsigned long long rng_seed = 0; // seed rng with time by default
    };

    /**
     * @brief Llama2 App.
     *
     */
    class Llama2
    {
    public:
        Llama2(const Llama2Config &config) : m_config(config), m_transformer(nullptr), m_tokenizer(nullptr), m_sampler(nullptr)
        {
            m_transformer = std::make_unique<Transformer>(m_config.checkpoint_path);
            if (m_config.steps == 0 || m_config.steps > m_transformer->getConfig().seq_len)
                m_config.steps = m_transformer->getConfig().seq_len; // override to ~max length
            m_tokenizer = std::make_unique<Tokenizer>(m_config.tokenizer_path, m_transformer->getConfig().vocab_size);
            m_sampler = std::make_unique<Sampler>(m_transformer->getConfig().vocab_size, m_config.temperature, m_config.topp, m_config.rng_seed);
        }

        ~Llama2()
        {
        }

        void generate(const std::string &prompt)
        {
            // encode the (string) prompt into tokens sequence
            int num_prompt_tokens = 0;

            Shape shape = {prompt.length() + 3};
            Tensor<CPU, int> prompt_tokens(shape);
            m_tokenizer->encode(prompt, 1, 0, prompt_tokens, num_prompt_tokens);

            if (num_prompt_tokens < 1)
            {
                std::cerr << "something is wrong, expected at least 1 prompt token" << std::endl;
                exit(EXIT_FAILURE);
            }

            // start the main loop
            long start = 0;               // used to time our code, only initialized after first iteration
            int next;                     // will store the next token in the sequence
            int token = prompt_tokens[0]; // kick off with the first token in the prompt
            int pos = 0;                  // position in the sequence
            while (pos < m_config.steps)
            {

                // forward the transformer to get logits for the next token
                float *logits = m_transformer->forward(token, pos);

                // advance the state machine
                if (pos < num_prompt_tokens - 1)
                {
                    // if we are still processing the input prompt, force the next prompt token
                    next = prompt_tokens[pos + 1];
                }
                else
                {
                    // otherwise sample the next token from the logits
                    next = m_sampler->sample(logits);
                }
                pos++;

                // data-dependent terminating condition: the BOS (=1) token delimits sequences
                if (next == 1)
                {
                    break;
                }

                // print the token as string, decode it with the Tokenizer object
                auto piece = m_tokenizer->decode(token, next);
                safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
                std::cout << std::flush;
                token = next;

                // init the timer here because the first iteration can be slower
                if (start == 0)
                {
                    start = time_in_ms();
                }
            }
            std::cout << std::endl;

            // report achieved tok/s (pos-1 because the timer starts after first iteration)
            if (pos > 1)
            {
                long end = time_in_ms();
                float64_t token_rate = (pos - 1) / static_cast<float64_t>(end - start) * 1000;
                std::cout << "acheived tok/s:" << token_rate << std::endl;
            }
        }

        void chat(const std::string &cli_user_prompt, const std::string &cli_system_prompt)
        {
            // I manually inspected the tokens for a few chat conversations compared to
            // python reference and that seemed ok, but this was not thoroughly tested and
            // is not safely implemented, it's more a proof of concept atm.

            // buffers for reading the system prompt and user prompt from stdin
            // you'll notice they are soomewhat haphazardly and unsafely set atm
            char system_prompt[512];
            char user_prompt[512];
            char rendered_prompt[1152];
            int num_prompt_tokens = 0;
            Shape shape = {1152};
            Tensor<CPU, int> prompt_tokens(shape);
            int user_idx;

            // start the main loop
            int8_t user_turn = 1; // user starts
            int next;             // will store the next token in the sequence
            int token;            // stores the current token to feed into the transformer
            int prev_token;
            int pos = 0; // position in the sequence
            while (pos < m_config.steps)
            {

                // when it is the user's turn to contribute tokens to the dialog...
                if (user_turn)
                {
                    // get the (optional) system prompt at position 0
                    if (pos == 0)
                    {
                        // at position 0, the user can also contribute a system prompt
                        if (cli_system_prompt.empty())
                        {
                            // system prompt was not passed in, attempt to get it from stdin
                            read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                        }
                        else
                        {
                            // system prompt was passed in, use it
                            strcpy(system_prompt, cli_system_prompt.c_str());
                        }
                    }
                    // get the user prompt
                    if (pos == 0 && cli_user_prompt.empty())
                    {
                        // user prompt for position 0 was passed in, use it
                        strcpy(user_prompt, cli_user_prompt.c_str());
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
                    m_tokenizer->encode(rendered_prompt, 1, 0, prompt_tokens, num_prompt_tokens);
                    user_idx = 0; // reset the user index
                    user_turn = 0;
                    std::cout << "Assistant: ";
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
                float *logits = m_transformer->forward(token, pos);
                next = m_sampler->sample(logits);
                pos++;

                if (user_idx >= num_prompt_tokens && next != 2)
                {
                    // the Assistant is responding, so print its output
                    auto piece = m_tokenizer->decode(token, next);
                    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
                    std::cout << std::flush;
                }
                if (next == 2)
                {
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }

    private:
        Llama2Config m_config;
        Transformer::ptr m_transformer;
        Tokenizer::ptr m_tokenizer;
        Sampler::ptr m_sampler;
    };

}
#endif