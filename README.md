
## llama2.rs 🦀

> Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure C? No? Well, now you can!

Great news! More stuff you didn't want but now can!

I've forked and ported Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) to Rust! 🦀 It's just as minimalistic as the original C code.
This is a quick-and-dirty first attempt.

Why? Because it was FUN! Plus, I'm curious to see how the C and Rust versions will evolve differently.

## How to run?
1. Grapb Karpathy's baby Llama2 ([Orig instructions](https://github.com/karpathy/llama2.c#feel-the-magic)) pretrained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset 

    ```bash
    wget https://karpathy.ai/llama2c/model.bin -P out
    ```
2. Make sure you have the tokenizer binary - `tokenizer.bin` (if not see [tokenizer.py](tokenizer.py))
3. Compile the Rust code
    ```bash
    cd llama2-rs
    cargo run --release
    ```

## Keeping up with the original
I'm pretty sure that `llama2.c` is going to move fast and get lots of contributions. 

So any contribution is welcome here!


## License
MIT
