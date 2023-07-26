## llama2.c

<img src="assets/llama_cute.jpg" width="300" height="300">

With the code in this repo you can train the Llama 2 LLM architecture from scratch in PyTorch, then export the weights to a binary file, and load that into one ~simple 500-line C file ([run.c](run.c)) that inferences the model. Alternatively, you can load, finetune, and inference Meta's Llama 2 (but this is still being actively fleshed out). Hence, this repo is a "fullstack" train + inference solution for Llama 2 LLM, with a focus on minimalism and simplicity. You might think that you need many billion parameter LLMs to do anything useful, but in fact very small LLMs can have surprisingly strong performance if you make the domain narrow enough. I recommend looking at the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) paper for inspiration.

Please note that this started recently as just a fun weekend project: I took my earlier [nanoGPT](https://github.com/karpathy/nanoGPT), tuned it to implement the Llama-2 architecture instead of GPT-2, and the meat of it was writing the C inference engine in [run.c](run.c). So the project is young and moving quickly. Hat tip to the awesome [llama.cpp](https://github.com/ggerganov/llama.cpp) for inspiring this project. I wanted something super minimal so I chose to hard-code the Llama 2 architecture, stick to fp32, and just roll one inference file of pure C with no dependencies.

## feel the magic

Let's just run a baby Llama 2 model in C. You need a model checkpoint. Download this 15M parameter model I trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (~58MB download) and place it into the default checkpoint directory `out`:

```bash
wget https://karpathy.ai/llama2c/model.bin -P out
```

(if that doesn't work try [google drive](https://drive.google.com/file/d/1aTimLdx3JktDXxcHySNrZJOOk8Vb1qBR/view?usp=share_link)). Compile and run the C code:

```bash
gcc -O3 -o run run.c -lm
./run out/model.bin
```

You'll see the text stream a sample. On my M1 MacBook Air this runs at ~110 tokens/s. See [performance](#performance) or the Makefile for compile flags that can significantly speed this up. Sample output:

> Once upon a time, there was a boy named Timmy. Timmy loved to play sports with his friends. He was very good at throwing and catching balls. One day, Timmy's mom gave him a new shirt to wear to a party. Timmy thought it was impressive and asked his mom to explain what a shirt could be for. "A shirt is like a special suit for a basketball game," his mom said. Timmy was happy to hear that and put on his new shirt. He felt like a soldier going to the army and shouting. From that day on, Timmy wore his new shirt every time he played sports with his friends at the party. Once upon a time, there was a little girl named Lily. She loved to play outside with her friends. One day, Lily and her friend Emma were playing with a ball. Emma threw the ball too hard and it hit Lily's face. Lily felt embarrassed and didn't want to play anymore.
> Emma asked Lily what was wrong, and Lily told her about her memory. Emma told Lily that she was embarrassed because she had thrown the ball too hard. Lily felt bad
> achieved tok/s: 129.146172

**Update**: I've now also uploaded a bigger checkpoint. This one is dim 512, 8 layers, 8 heads and context length 1024, a ~44M param Transformer. It trained for 200K iterations batch size 32 on 4XA100 40GB GPUs in ~8 hours. You can use this bigger and more powerful checkpoint like so:

```bash
wget https://karpathy.ai/llama2c/model44m.bin -P out44m
./run out44m/model44m.bin
```

This still runs at interactive rates and samples more coherent and diverse stories:

> Once upon a time, there was a little girl named Lily. She loved playing with her toys on top of her bed. One day, she decided to have a tea party with her stuffed animals. She poured some tea into a tiny teapot and put it on top of the teapot. Suddenly, her little brother Max came into the room and wanted to join the tea party too. Lily didn't want to share her tea and she told Max to go away. Max started to cry and Lily felt bad. She decided to yield her tea party to Max and they both shared the teapot. But then, something unexpected happened. The teapot started to shake and wiggle. Lily and Max were scared and didn't know what to do. Suddenly, the teapot started to fly towards the ceiling and landed on the top of the bed. Lily and Max were amazed and they hugged each other. They realized that sharing was much more fun than being selfish. From that day on, they always shared their tea parties and toys.

**Update 2**: The 110M param model is also available now, see [models](#models).


## Meta's Llama 2 models

As the neural net architecture is identical, we can also inference the Llama 2 models released by Meta. Sadly there is a bit of friction here due to licensing (I can't directly upload the checkpoints, I think). So Step 1, get the Llama 2 checkpoints by following the [Meta instructions](https://github.com/facebookresearch/llama). Once we have those checkpoints, we have to convert them into the llama2.c format. For this we use the `export_meta_llama_bin.py` file, e.g. for 7B model:

```bash
python export_meta_llama_bin.py path/to/llama/model/7B llama2_7b.bin
```

The export will take ~10 minutes or so and generate a 26GB file (the weights of the 7B model in float32) called `llama2_7b.bin` in the current directory. It has been [reported](https://github.com/karpathy/llama2.c/pull/85) that despite efforts, the 13B export currently doesn't work for unknown reaons (accepting PRs for fix). We can run the model as normal:

```bash
./run llama2_7b.bin
```

This ran at about 4 tokens/s compiled with OpenMP on 96 threads on my CPU Linux box in the cloud. (On my MacBook Air M1, currently it's closer to 30 seconds per token if you just build with `make runfast`.) Example output:

> The purpose of this document is to highlight the state-of-the-art of CoO generation technologies, both recent developments and those in commercial use. The focus is on the technologies with the highest merit to become the dominating processes of the future and therefore to be technologies of interest to S&amp;T ... R&amp;D. As such, CoO generation technologies developed in Russia, Japan and Europe are described in some depth. The document starts with an introduction to cobalt oxides as complex products and a short view on cobalt as an essential material. The document continues with the discussion of the available CoO generation processes with respect to energy and capital consumption as well as to environmental damage.

base models... ¯\\_(ツ)_/¯. Since we can inference the base model, it should be possible to also inference the chat model quite easily, and have a conversation with it. And if we can find a way to run 7B more efficiently, we can start adding LoRA to our training script, and going wild with finetunes all within the repo!

## models

For the sake of examples of smaller, from-scratch models, I trained multiple models on TinyStories and catalogue them here:

| model | dim | n_layers | n_heads | max context length | parameters | val loss | download
| --- | --- | --- | --- | --- | --- | --- | --- |
| OG | 288 | 6 | 6 | 256 | 15M | | [model.bin](https://karpathy.ai/llama2c/model.bin) |
| 44M| 512 | 8 | 8 | 1024 | 44M | | [model44m.bin](https://karpathy.ai/llama2c/model44m.bin) |
| 110M| 768 | 12 | 12 | 1024 | 110M | 0.7601 | [model110m.bin](https://karpathy.ai/llama2c/model110m.bin) |

You'll notice that the 110M model is equivalent to GPT-1 in size. Alternatively, this is also the smallest model in the GPT-2 series (`GPT-2 small`), except the max context length is only 1024 instead of 2048. The only notable changes from GPT-1/2 architecture is that Llama uses RoPE relatively positional embeddings instead of absolute/learned positional embeddings, a bit more fancy SwiGLU non-linearity in the MLP, RMSNorm instead of LayerNorm, bias=False on all Linear layers, and is optionally multiquery (but this is not yet supported in llama2.c).

## training

Let's see how we can train a baby Llama 2 from scratch using the code in this repo. First let's download and pretokenize some source dataset, e.g. I like [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) so this is the only example currently available in this repo. But it should be very easy to add datasets, see the code.

```bash
python tinystories.py download
python tinystories.py pretokenize
```

Then train our model:

```bash
python train.py
```

See the train.py script for more exotic launches and hyperparameter overrides. I didn't tune the hyperparameters, I expect simple hyperparameter exploration should give better models. Totally understand if you want to skip model training, for simple demo just download my pretrained model and save it into the directory `out`:

```bash
wget https://karpathy.ai/llama2c/model.bin -P out
```

Once we have the model.bin file, we can inference in C. Compile the C code first:

```bash
gcc -O3 -o run run.c -lm
```

You can now run it simply as

```bash
./run out/model.bin
```

Watch the tokens stream by, fun! We can also run the PyTorch inference script for comparison (to run, add [model.ckpt](https://drive.google.com/file/d/1SM0rMxzy7babB-v4MfTg1GFqOCgWar5w/view?usp=share_link) to /out if you haven't already):

```bash
python sample.py
```

Which gives the same results. More detailed testing will be done in `test_all.py`, run as:

```bash
$ pytest
```

Currently you will need two files to test or sample: the [model.bin](https://drive.google.com/file/d/1aTimLdx3JktDXxcHySNrZJOOk8Vb1qBR/view?usp=share_link) file and the [model.ckpt](https://drive.google.com/file/d/1SM0rMxzy7babB-v4MfTg1GFqOCgWar5w/view?usp=share_link) file from PyTorch training I ran earlier. I have to think through running the tests without having to download 200MB of data.

## performance

*(NOTE: this guide is not great because I personally spend a lot of my time in Python land and don't have an amazing understanding of a lot of these features and flags. If someone does and is willing to help document and briefly describe some of these and their tradeoffs, I'd welcome a PR)*

There are many ways to potentially speed up this code depending on your system. Here we document a few together with a high-level guide on what they do. Here's again the default way to compile, but using -O3:

```bash
gcc -O3 -o run run.c -lm
```

-O3 includes optimizations that are expensive in terms of compile time and memory usage. Including vectorization, loop unrolling, and predicting branches. Here's a few more to try.

`-Ofast` Run additional optimizations which may break compliance with the C/IEEE specifications, in addition to `-O3`. See [the GCC docs](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) for more information.

`-march=native` Compile the program to use the architecture of the machine you're compiling on rather than a more generic CPU. This may enable additional optimizations and hardware-specific tuning such as improved vector instructions/width.

The fastest throughput I saw so far on my MacBook Air (M1) is with:

```bash
gcc -Ofast -o run run.c -lm
```

You can also experiment with replacing `gcc` with `clang`.

**OpenMP** Big improvements can also be achieved by compiling with OpenMP, which "activates" the `#pragma omp parallel for` inside the matmul and attention. You can compile e.g. like so:

```bash
clang -Ofast -fopenmp -march=native run.c  -lm  -o run
```

You can try swapping clang/gcc, and may try to leave out -march=native. However, when you run inference make sure to use OpenMP flags to set the number of threads, e.g.:

```bash
OMP_NUM_THREADS=4 ./run out/model.bin
```

Depending on your system resources you may want to tweak these hyperparameters. (TODO: I am not intimately familiar with OpenMP and its configuration, if someone would like to flesh out this section I would welcome a PR).

## unsorted todos

- why is there a leading space in C sampling code when we `./run`?
- support Llama 2 Chat models, and tune run.c to Chat UI/UX
- possibly include emscripten / web backend (as seen in @gg PR)
- currently the project only runs in fp32, want to explore more reduced precision inference.
- todo multiquery support? doesn't seem as useful for smaller models that run on CPU (?)
- todo support inferencing beyond max_seq_len steps, have to think through the kv cache
- why is MFU so low (~10%) on my A100 40GB for training?
- weird errors with torch.compile and wandb when using DDP
- (LoRA) finetuning of Llama 2 models
- make more better tests to decrease yolo

## ack

I trained the llama2.c storyteller models on a 4X A100 40GB box graciously provided by the excellent [Lambda labs](https://lambdalabs.com/service/gpu-cloud), thank you.

## discord

Figured it's possible to reuse my existing discord channel (that I use for my [zero to hero youtube series](https://karpathy.ai/zero-to-hero.html)), see #llama2c channel on [discord](https://discord.gg/3zy8kqD9Cp), for any quick questions, related discussions, etc.

## License

MIT
