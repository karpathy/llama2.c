#define TESTING
#include "run.c"

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

#define TEST_PROMPT_ENCODINGS_TEST_CASES \
    TEST_IMPL("", 1) \
    TEST_IMPL("I believe the meaning of life is", 1, 306, 4658, 278, 6593, 310, 2834, 338) \
    TEST_IMPL("Simply put, the theory of relativity states that ", 1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871) \
    TEST_IMPL("A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ", 1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871) \
    TEST_IMPL("Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>", 1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923, 968, 1149)

void test_prompt_encodings() {
    // let's verify that the Tokenizer works as expected

    char *tokenizer_path = "tokenizer.bin";
    int vocab_size = 32000;
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, vocab_size);

#define TEST_IMPL(p, ...) \
    do {\
        char *prompt = p; \
        int expected_tokens[] = {__VA_ARGS__}; \
        test_prompt_encoding(&tokenizer, prompt, expected_tokens, sizeof(expected_tokens) / sizeof(int)); \
    } while(0);

    TEST_PROMPT_ENCODINGS_TEST_CASES
#undef TEST_IMPL

    // memory and file handles cleanup
    free_tokenizer(&tokenizer);
}

int main(int argc, char *argv[]) {
    test_prompt_encodings();
    printf("ALL OK\n");
}
