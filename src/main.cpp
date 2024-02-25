#include <iostream>
#include <cstdio>
#include <cstring>
#include <llama2cpp/sampler.hpp>
#include <llama2cpp/tokenizer.hpp>
#include <llama2cpp/transformer.hpp>
#include <llama2cpp/llama2.hpp>

void error_usage()
{
    std::cout << "Usage:   run <checkpoint> [options]\n"
              << std::endl;
    std::cout << "Example: run model.bin -n 256 -i \"Once upon a time\"\n"
              << std::endl;
    std::cout << "Options:\n"
              << std::endl;
    std::cout << "  -t <float>  temperature in [0,inf], default 1.0\n"
              << std::endl;
    std::cout << "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n"
              << std::endl;
    std::cout << "  -s <int>    random seed, default time(NULL)\n"
              << std::endl;
    std::cout << "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n"
              << std::endl;
    std::cout << "  -i <string> input prompt\n"
              << std::endl;
    std::cout << "  -z <string> optional path to custom tokenizer\n"
              << std::endl;
    std::cout << "  -m <string> mode: generate|chat, default: generate\n"
              << std::endl;
    std::cout << "  -y <string> (optional) system prompt in chat mode\n"
              << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    // default parameters
    char *checkpoint_path = NULL; // e.g. out/model.bin
    char *tokenizer_path = "../tokenizer.bin";
    float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;                 // number of steps to run for
    char *prompt = NULL;             // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";         // generate|chat
    char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2)
    {
        checkpoint_path = argv[1];
    }
    else
    {
        error_usage();
    }
    for (int i = 2; i < argc; i += 2)
    {
        // do some basic validation
        if (i + 1 >= argc)
        {
            error_usage();
        } // must have arg after flag
        if (argv[i][0] != '-')
        {
            error_usage();
        } // must start with dash
        if (strlen(argv[i]) != 2)
        {
            error_usage();
        } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't')
        {
            temperature = atof(argv[i + 1]);
        }
        else if (argv[i][1] == 'p')
        {
            topp = atof(argv[i + 1]);
        }
        else if (argv[i][1] == 's')
        {
            rng_seed = atoi(argv[i + 1]);
        }
        else if (argv[i][1] == 'n')
        {
            steps = atoi(argv[i + 1]);
        }
        else if (argv[i][1] == 'i')
        {
            prompt = argv[i + 1];
        }
        else if (argv[i][1] == 'z')
        {
            tokenizer_path = argv[i + 1];
        }
        else if (argv[i][1] == 'm')
        {
            mode = argv[i + 1];
        }
        else if (argv[i][1] == 'y')
        {
            system_prompt = argv[i + 1];
        }
        else
        {
            error_usage();
        }
    }

    // parameter validation/overrides
    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0)
        temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp)
        topp = 0.9;
    if (steps < 0)
        steps = 0;

    // build the Transformer via the model .bin file
    llama2cpp::Transformer transformer;
    llama2cpp::build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    llama2cpp::Tokenizer tokenizer;
    llama2cpp::build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    llama2cpp::Sampler sampler;
    llama2cpp::build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0)
    {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    }
    else if (strcmp(mode, "chat") == 0)
    {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    }
    else
    {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    llama2cpp::free_sampler(&sampler);
    llama2cpp::free_tokenizer(&tokenizer);
    llama2cpp::free_transformer(&transformer);
    return 0;
}