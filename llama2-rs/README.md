I've ported Karpathy's llama2.c to Rust! 🦀 It's just as minimalistic as the original C code.
This is a quick-and-dirty first attempt.

Why? Because it was FUN! Plus, I'm curious to see how the C and Rust versions will evolve differently.

Performance is on par with C (w/o -Ofast). Benchmarked on my dev VM.


Follow original instructions for getting model binary and tokenizer (for now paths are hardcoded)

```bash
cd llama2-rs
cargo run --release
```
