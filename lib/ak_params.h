#ifndef _ak_params_H
#define _ak_params_H

void ak_params_error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

typedef struct {
    char *checkpoint_path;
    char *tokenizer_path;
    float temperature;
    float topp;
    int steps;
    char *prompt;
    unsigned long long rng_seed;
    char *mode;
    char *system_prompt;
} ak_params_t;

ak_params_t* ak_params_init(int argc, char *argv[]) {
    ak_params_t* p = (ak_params_t*)malloc(sizeof(*p));
    p->checkpoint_path = NULL;  // e.g. out/model.bin
    p->tokenizer_path = (char *)"tokenizer.bin";
    p->temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    p->topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    p->steps = 256;            // number of steps to run for
    p->prompt = NULL;        // prompt string
    p->rng_seed = 0; // seed rng with time by default
    p->mode = (char *)"generate";    // generate|chat
    p->system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { p->checkpoint_path = argv[1]; } else { ak_params_error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { ak_params_error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { ak_params_error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { ak_params_error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { p->temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { p->topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { p->rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { p->steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { p->prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { p->tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { p->mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { p->system_prompt = argv[i + 1]; }
        else { ak_params_error_usage(); }
    }

    // parameter validation/overrides
    if (p->rng_seed <= 0) p->rng_seed = (unsigned int)time(NULL);
    if (p->temperature < 0.0) p->temperature = 0.0;
    if (p->topp < 0.0 || 1.0 < p->topp) p->topp = 0.9;
    if (p->steps < 0) p->steps = 0;
    return p;
}

void ak_params_destroy(ak_params_t* p) {
    free(p);
}

#endif