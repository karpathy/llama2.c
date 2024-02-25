#include <iostream>
#include <cstdio>
#include <cstring>
#include <llama2cpp/sampler.hpp>
#include <llama2cpp/tokenizer.hpp>
#include <llama2cpp/transformer.hpp>
#include <llama2cpp/llama2.hpp>

struct ConsoleArgs
{
    // default parameters
    std::string checkpoint_path = ""; // e.g. out/model.bin
    std::string tokenizer_path = "../tokenizer.bin";
    float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;                 // number of steps to run for
    std::string prompt = "";         // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    std::string mode = "generate";   // generate|chat
    std::string system_prompt = "";  // the (optional) system prompt to use in chat mode

    static void error_usage()
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

    void parse(int argc, char **argv)
    {
        // poor man's C++ argparse so we can override the defaults above from the command line
        if (argc >= 2)
        {
            checkpoint_path = std::string(argv[1]);
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
                prompt = std::string(argv[i + 1]);
            }
            else if (argv[i][1] == 'z')
            {
                tokenizer_path = std::string(argv[i + 1]);
            }
            else if (argv[i][1] == 'm')
            {
                mode = std::string(argv[i + 1]);
            }
            else if (argv[i][1] == 'y')
            {
                system_prompt = std::string(argv[i + 1]);
            }
            else
            {
                error_usage();
            }
        }
    }
};

int main(int argc, char **argv)
{

    ConsoleArgs args;
    args.parse(argc, argv);

    llama2cpp::Llama2Config config;
    config.checkpoint_path = args.checkpoint_path;
    config.tokenizer_path = args.tokenizer_path;
    config.temperature = args.temperature;
    config.topp = args.topp;
    config.steps = args.steps;
    config.rng_seed = args.rng_seed;

    llama2cpp::Llama2 llama2(config);

    // run!
    if (args.mode == "generate")
    {
        llama2.generate(args.prompt);
    }
    else if (args.mode == "chat")
    {
        llama2.chat(args.prompt, args.system_prompt);
    }
    else
    {
        std::cerr << "unknown mode" << args.mode << std::endl;
        ConsoleArgs::error_usage();
    }
}