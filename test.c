#define TESTING
#include "run.c"

#include <riscv_vector.h>

void assert_eq(int a, int b) {
    if (a != b) {
        printf("Assertion failed: %d != %d\n", a, b);
        exit(EXIT_FAILURE);
    }
}

void test_prompt_encoding(Tokenizer* tokenizer, char* prompt, int* expected_tokens, int num_expected_tokens) {
    // encode
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    int num_prompt_tokens = 0; // the total number of prompt tokens
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    #if VERBOSITY == 1
    // print maybe
    printf("expected tokens:\n");
    for (int i = 0; i < num_expected_tokens; i++) printf("%d ", expected_tokens[i]);
    printf("\n");
    printf("actual tokens:\n");
    for (int i = 0; i < num_prompt_tokens; i++) printf("%d ", prompt_tokens[i]);
    printf("\n");
    #endif

    // verify
    assert_eq(num_prompt_tokens, num_expected_tokens);
    for (int i = 0; i < num_prompt_tokens; i++) {
        assert_eq(prompt_tokens[i], expected_tokens[i]);
    }

    #if VERBOSITY == 1
    printf("OK\n");
    printf("---\n");
    #endif
    free(prompt_tokens);
}

void test_prompt_encodings() {
    // let's verify that the Tokenizer works as expected

    char *tokenizer_path = "tokenizer.bin";
    int vocab_size = 32000;
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, vocab_size);

    // test 0 (test the empty string) (I added this as a simple case)
    char *prompt0 = "";
    int expected_tokens0[] = {1};
    test_prompt_encoding(&tokenizer, prompt0, expected_tokens0, sizeof(expected_tokens0) / sizeof(int));

    // the tests below are taken from the Meta Llama 2 repo example code
    // https://github.com/facebookresearch/llama/blob/main/example_text_completion.py
    // and the expected tokens come from me breaking in the debugger in Python

    // test 1
    char *prompt = "I believe the meaning of life is";
    int expected_tokens[] = {1, 306, 4658, 278, 6593, 310, 2834, 338};
    test_prompt_encoding(&tokenizer, prompt, expected_tokens, sizeof(expected_tokens) / sizeof(int));

    // test 2
    char* prompt2 = "Simply put, the theory of relativity states that ";
    int expected_tokens2[] = {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871};
    test_prompt_encoding(&tokenizer, prompt2, expected_tokens2, sizeof(expected_tokens2) / sizeof(int));

    // test 3
    char* prompt3 = "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ";
    int expected_tokens3[] = {1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871};
    test_prompt_encoding(&tokenizer, prompt3, expected_tokens3, sizeof(expected_tokens3) / sizeof(int));

    // test 4
    char* prompt4 = "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>";
    int expected_tokens4[] = {1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923, 968, 1149};
    test_prompt_encoding(&tokenizer, prompt4, expected_tokens4, sizeof(expected_tokens4) / sizeof(int));

    // memory and file handles cleanup
    free_tokenizer(&tokenizer);
}

// source: https://github.com/opencv/opencv/blob/ae21368eb9b66b448effc60247be8d83056ade80/cmake/checks/cpu_rvv.cpp
int test_rvv()
{
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    unsigned long ptr[2] = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
    vuint8m1_t a = vreinterpret_v_u64m1_u8m1(vle64_v_u64m1(ptr, 2));
    vfloat32m1_t val = vle32_v_f32m1((const float*)(src), 4);
    return (int)vfmv_f_s_f32m1_f32(val);
}



void test_generate(char* prompt, char* checkpoint_path, float temperature, int steps, float topp, const char* expected){
    long start_g = time_in_ms();
    char *tokenizer_path = "tokenizer.bin";
    unsigned long long rng_seed = 124; // seed rng with time by default

    // parameter validation/overrides
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!

    freopen("output.txt", "wt", stdout);  // redirect output
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    freopen("/dev/tty", "w", stdout);  // resume

        
    long start4 = time_in_ms();
    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    long end4 = time_in_ms();
    fprintf(stderr, "memory and file handles cleanup ms: %f\n", (double)(end4-start4));

    long start5 = time_in_ms();
    // Check
    FILE* f = fopen("output.txt", "rt");
    fseek(f, 0, SEEK_END);
    const size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char output[sz];
    fread(output, sizeof(char), sz, f);
    output[sz - 1] = '\0';
    fclose(f);
    long end5 = time_in_ms();
    fprintf(stderr, "check ms: %f\n", (double)(end5-start5));

    int res = strcmp(expected, output);
    if (res != 0) {
        printf("Expected: %s\n\nGenerated: %s\n", expected, output);
    }
    assert_eq(res, 0);

    long end_g = time_in_ms();
    fprintf(stderr, "time ms: %f\n\n", (double)(end_g-start_g));
}

int main(int argc, char *argv[]) {
    test_prompt_encodings();
    test_rvv();

    const char* expected = "That was the darkest day of the year. The stars were shining bright in the sky and the birds were singing.\n\
\"Mommy, why is it so dark?\" asked the little girl, pointing out her finger.\n\
\"Well, the sun is setting and it will be a beautiful night,\" replied her mom.\n\
The little girl looked up at the sky and smiled. \"I like it when the sun sets,\" she said.\n\
\"I know, sweetie. The";
    test_generate("That was the darkest day of the year.", "stories15M.bin", 0.7f, 100, 0.9f, expected);

const char* expected2="It was dark and cold around. The little girl was feeling scared. She looked around and saw a big, dark room. She wanted to go in, but she was too scared.\n\
Suddenly, she heard a noise. It was coming from the corner of the room. She slowly walked over and saw a big, black cat. It was meowing and seemed to be trying to get her attention.\n\
The little girl was still scared, but she was also curious. She";

    test_generate ("It was dark and cold around.", "stories110M.bin", 0.3f, 103, 0.6f, expected2);
    printf("ALL OK\n");

}
