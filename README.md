## llama2.dart

This is a fork of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), implemented in (Almost) Pure Dart, except for some args parsing utility library.

### To run :

Instal Dart

```bash
brew tap dart-lang/dart
brew install dart
```

Install the arg parsing dependency

```bash
dart pub add args
```

Download the dataset:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

```bash
dart run run.dart -c ./stories15M.bin -i "PROMPT GOES HERE"
```

## Performance

Dart suprisingly ok performance being a single threaded language, tho it's starting to struggle at 110M:
Tested on M2 Max Chip

| Model | Token/s      |
| ----- | ------------ |
| 15M   | tok/s: 17.78 |
| 42M   | tok/s: 6.43  |
| 110M  | tok/s: 2.47  |

### Original README

Extract from the original Repo:

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

Train the Llama 2 LLM architecture in PyTorch then inference it with one simple 700-line C file ([run.c](run.c)). You might think that you need many billion parameter LLMs to do anything useful, but in fact very small LLMs can have surprisingly strong performance if you make the domain narrow enough (ref: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) paper). This repo is a "fullstack" train + inference solution for Llama 2 LLM, with focus on minimalism and simplicity.

As the architecture is identical, you can also load and inference Meta's Llama 2 models. However, the current code only inferences models in fp32, so you will most likely not be able to productively load models larger than 7B. Work on model quantization is currently ongoing.

Please note that this repo started recently as a fun weekend project: I took my earlier [nanoGPT](https://github.com/karpathy/nanoGPT), tuned it to implement the Llama-2 architecture instead of GPT-2, and the meat of it was writing the C inference engine in [run.c](run.c). So the project is young and moving quickly. Hat tip to the awesome [llama.cpp](https://github.com/ggerganov/llama.cpp) for inspiring this project. Compred to llama.cpp, I wanted something super simple, minimal, and educational so I chose to hard-code the Llama 2 architecture and just roll one inference file of pure C with no dependencies.

Please refer to [Original README](/ORIGINAL.md) or the upstream repo for more information on llama2.c
