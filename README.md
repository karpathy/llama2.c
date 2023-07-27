
## llama2.rs 🦀

> Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure C? No? Well, now you can!

Great news! More stuff you didn't want but now can!

I've forked and ported Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) to Rust! 🦀 It's just as minimalistic as the original C code.
This is a quick-and-dirty first attempt.

Why? Because it was FUN! Plus, I'm curious to see how the C and Rust versions will evolve differently.

## How to run?
1. Grab Karpathy's baby Llama2 ([Orig instructions](https://github.com/karpathy/llama2.c#feel-the-magic)) pretrained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset 

    ```bash
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
    ```
2. Make sure you have the tokenizer binary - `tokenizer.bin` (if not see [tokenizer.py](tokenizer.py))
3. Compile and run the Rust code

    Single threaded:

    ```bash
    cargo run --release stories42M.bin 0.9  # <model_path> [temperature]
    ```

    Multipthreaded (depends on Rayon)
    ```bash
    cargo run --release  -F parallel stories42M.bin 0.9 # <model_path> [temperature]
    ```

    You can also run `make rust` or `make rustfast` to get `run-rs` binary 
## Performance

Hacky tokens/sec measurement on dev VM (16 cores/64G mem).

|    tok/s   | 15M | 42M | 110M
|-------|-----|-----|-----|
| 1 core|  ~75|   ~25   | ~10
| 12 cores |  ~310   |  ~110   | ~50



## Keeping up with the original
I'm pretty sure that `llama2.c` is going to move fast and get lots of contributions. 

So any contribution is welcome here!

### Contribution Ideas
- Parallelize attention over heads
- WASM port?


## License
MIT
