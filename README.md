## llama2.c

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure C? No? Well, now you can!

Train the Llama 2 LLM architecture in PyTorch then inference it with one simple 700-line C file ([run.c](run.c)). You might think that you need many billion parameter LLMs to do anything useful, but in fact very small LLMs can have surprisingly strong performance if you make the domain narrow enough (ref: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) paper). This repo is a "fullstack" train + inference solution for Llama 2 LLM, with focus on minimalism and simplicity.

As the architecture is identical, you can also load and inference Meta's Llama 2 models. However, the current code only inferences models in fp32, so you will most likely not be able to productively load models larger than 7B. Work on model quantization is currently ongoing.

Please note that this repo started recently as a fun weekend project: I took my earlier [nanoGPT](https://github.com/karpathy/nanoGPT), tuned it to implement the Llama-2 architecture instead of GPT-2, and the meat of it was writing the C inference engine in [run.c](run.c). So the project is young and moving quickly. Hat tip to the awesome [llama.cpp](https://github.com/ggerganov/llama.cpp) for inspiring this project. Compared to llama.cpp, I wanted something super simple, minimal, and educational so I chose to hard-code the Llama 2 architecture and just roll one inference file of pure C with no dependencies.

## feel the magic

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karpathy/llama2.c/blob/master/run.ipynb)

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone https://github.com/karpathy/llama2.c.git
```

Then, open the repository folder:

```bash
cd llama2.c
```

Now, let's just run a baby Llama 2 model in C. You need a model checkpoint. Download this 15M parameter model I trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (~60MB download):

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Compile and run the C code:

```bash
make run
./run stories15M.bin
```

You'll see the text stream a sample. On my M1 MacBook Air this runs at ~110 tokens/s. See [performance](#performance) or the Makefile for compile flags that can significantly speed this up. We can also try a bit bigger 42M parameter model:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
./run stories42M.bin
```

This still runs at interactive rates and samples more coherent and diverse stories:

> Once upon a time, there was a little girl named Lily. She loved playing with her toys on top of her bed. One day, she decided to have a tea party with her stuffed animals. She poured some tea into a tiny teapot and put it on top of the teapot. Suddenly, her little brother Max came into the room and wanted to join the tea party too. Lily didn't want to share her tea and she told Max to go away. Max started to cry and Lily felt bad. She decided to yield her tea party to Max and they both shared the teapot. But then, something unexpected happened. The teapot started to shake and wiggle. Lily and Max were scared and didn't know what to do. Suddenly, the teapot started to fly towards the ceiling and landed on the top of the bed. Lily and Max were amazed and they hugged each other. They realized that sharing was much more fun than being selfish. From that day on, they always shared their tea parties and toys.

You can also prompt the model with a prefix or a number of additional command line arguments, e.g. to sample at temperature 0.8 for 256 steps and with a prompt:

```bash
./run stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
```

> One day, Lily met a Shoggoth. He was very shy, but was also very generous. Lily said “Hello Shoggy! Can I be your friend?” Shoggy was happy to have a friend and said “Yes, let’s explore the universe together!” So they set off on a journey to explore the universe. As they travelled, Shoggy was happy to explain to Lily about all the wonderful things in the universe. At the end of the day, Lily and Shoggy had gathered lots of wonderful things from the universe, and they both felt very proud. They promised to explore the universe as one big pair and to never stop being generous to each other.

There is also an even better 110M param model available, see [models](#models).

Quick note on sampling, the recommendation for ~best results is to sample with `-t 1.0 -p 0.9`, i.e. temperature 1.0 (default) but also top-p sampling at 0.9 (default). Intuitively, top-p ensures that tokens with tiny probabilities do not get sampled, so we can't get "unlucky" during sampling, and we are less likely to go "off the rails" afterwards. More generally, to control the diversity of samples use either the temperature (i.e. vary `-t` between 0 and 1 and keep top-p off with `-p 0`) or the top-p value (i.e. vary `-p` between 0 and 1 and keep `-t 1`), but not both. Nice explainers on LLM sampling strategies include [this](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/), [this](https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p) or [this](https://huggingface.co/blog/how-to-generate).

## Meta's Llama 2 models

As the neural net architecture is identical, we can also inference the Llama 2 models released by Meta. Sadly there is a bit of friction here due to licensing (I can't directly upload the checkpoints, I think). So Step 1, get the Llama 2 checkpoints by following the [Meta instructions](https://github.com/facebookresearch/llama). Once we have those checkpoints, we have to convert them into the llama2.c format.
For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export.py` file, e.g. for 7B model:

```bash
python export.py llama2_7b.bin --meta-llama path/to/llama/model/7B
```

The export will take ~10 minutes or so and generate a 26GB file (the weights of the 7B model in float32) called `llama2_7b.bin` in the current directory. It has been [reported](https://github.com/karpathy/llama2.c/pull/85) that despite efforts. I would not attempt to run anything above 7B right now for two reasons: first, 13B+ currently doesn't work because of integer flow in pointer arithmetic, which is yet to be fixed, and second, even if it were fixed, this repo is doing float32 inference right now, so it would be fairly unusably slow. Once the export is done, we can run it:

```bash
./run llama2_7b.bin
```

This ran at about 4 tokens/s compiled with [OpenMP](#OpenMP) on 96 threads on my CPU Linux box in the cloud. (On my MacBook Air M1, currently it's closer to 30 seconds per token if you just build with `make runfast`.) Example output:

> The purpose of this document is to highlight the state-of-the-art of CoO generation technologies, both recent developments and those in commercial use. The focus is on the technologies with the highest merit to become the dominating processes of the future and therefore to be technologies of interest to S&amp;T ... R&amp;D. As such, CoO generation technologies developed in Russia, Japan and Europe are described in some depth. The document starts with an introduction to cobalt oxides as complex products and a short view on cobalt as an essential material. The document continues with the discussion of the available CoO generation processes with respect to energy and capital consumption as well as to environmental damage.

base models... ¯\\_(ツ)_/¯. Since we can inference the base model, it should be possible to also inference the chat model quite easily, and have a conversation with it. And if we can find a way to run 7B more efficiently, we can start adding LoRA to our training script, and going wild with finetunes all within the repo!

You can also chat with the Llama Chat models. Export the chat model exactly as above:

```bash
python export.py llama2_7b_chat.bin --meta-llama /path/to/7B-chat
```

Then chat with it by specifying the chat mode using the `-m` flag, e.g.:

```bash
./run llama2_7b_chat.bin -m chat
```

You can also try Meta's Code Llama models even if support for them is incomplete. In particular, some hyperparameters changed (e.g. the constant in RoPE layer), so the inference is not exactly correct and a bit buggy right now. Looking into fixes. Make sure to build the tokenizer for the plain and instruct variants and pass it when doing inference.

```bash
python export.py codellama2_7b.bin --meta-llama /path/to/CodeLlama-7b
python tokenizer.py --tokenizer-model=/path/to/CodeLlama-7b/tokenizer.model
./run codellama2_7b.bin -z /path/to/CodeLlama-7b/tokenizer.bin
```

Chat with Code Llama Instruct:

```bash
python export.py codellama2_7b_instruct.bin --meta-llama /path/to/CodeLlama-7b-Instruct
python tokenizer.py --tokenizer-model=/path/to/CodeLlama-7b-Instruct/tokenizer.model
./run codellama2_7b_instruct.bin -m chat -z /path/to/CodeLlama-7b-Instruct/tokenizer.bin
```

## int8 quantization

The (default) script [run.c](run.c), above, uses a float32 forward pass, where the entire calculation of the forward pass is kept in fp32. This is very easy to understand as far as reference code goes, but it has the following downsides: the model checkpoint files are very large (it takes 4 bytes per every individual weight), and the forward pass is relatively slow. The (very) common inference optimization employed in practice is to quantize the model parameters to lower precision, giving up a little bit of correctness in return for smaller checkpoint sizes and faster forward passes (as most of the inference uses integer arithmetic). Empirically, LLMs can tolerate precisions as low as 4-bit (or even lower), but we use int8 here because it is a "safe" setting that gets us the benefits but doesn't sacrifice too much of the model accuracy. Only the weights that participate in matmuls are quantized. All the other parameters (e.g. especially the scale and bias in RMSNorm) are kept in float32, because these layers are very sensitive. Now, if all you're after is reduction in checkpoint sizes, you could quantize the weights, save the checkpoint, and then dequantize them in run.c, and do float32 inference as normal and call it a day. This is totally fine. But here, we go one step further (as is standard practice) and additionally quantize the activations in the forward pass. This requires us to dynamically quantize and dequantize between float32 and int8 at runtime, which adds overhead. But the benefit is that now the majority of the calculations (the matmuls especially!) are using pure integer arithmetic, where both weights and activations enter as int8. This is where the speedups can fundamentally come from. The version we use is the "Q8_0" quantization (llama.cpp terminology), where the 0 means that the weight quantization is symmetric around 0, quantizing to the range [-127, 127].

The quantized forward pass is implemented in [runq.c](runq.c). To use it, we have to export the model in the quantized format. For example, the float32 version of Llama 2 7B was exported as:

```
python export.py llama2_7b.bin --meta-llama path/to/llama/model/7B
```

This creates a 26GB file, because each one of 7B parameters is 4 bytes (fp32). To export it quantized, we instead use version 2 export:

```
python export.py llama2_7b_q80.bin --version 2 --meta-llama path/to/llama/model/7B
```

This runs for a few minutes, but now creates only a 6.7GB file. For exporting non-meta checkpoints you would use the --checkpoint arg instead of --meta-llama arg (more docs on this later, below). Now let's inference them. I like to use OMP here because these are big models, so e.g. on my Linux box:

```
make runomp
OMP_NUM_THREADS=64 ./run llama2_7b.bin -n 40
OMP_NUM_THREADS=64 ./runq llama2_7b_q80.bin -n 40
```

This runs 40 steps just to get a timing. The float32 version for me runs at 4.6 tok/s, and the int8 version at 14 tok/s. So we achieved a 3X speedup while reducing the checkpoint size by 4X. However, the forward pass is quantized to int8, and therefore silently very slightly lower quality.

## huggingface models

We can load any huggingface models that use the Llama 2 architecture. See the script [export.py](export.py) and the `--hf` flag to export the model .bin file.

## models

For the sake of examples of smaller, from-scratch models, I trained a small model series on TinyStories. All of these trained in a few hours on my training setup (4X A100 40GB GPUs). The 110M took around 24 hours. I am hosting them on huggingface hub [tinyllamas](https://huggingface.co/karpathy/tinyllamas), both in the original PyTorch .pt, and also in the llama2.c format .bin:

| model | dim | n_layers | n_heads | n_kv_heads | max context length | parameters | val loss | download
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 260K | 64 | 5 | 8 | 4 | 512 | 260K | 1.297 | [stories260K](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K)
| OG | 288 | 6 | 6 | 6 | 256 | 15M | 1.072 | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) |
| 42M| 512 | 8 | 8 | 8 | 1024 | 42M | 0.847 | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) |
| 110M| 768 | 12 | 12 | 12 | 1024 | 110M | 0.760 | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) |

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

**brief training guide**. See the train.py script for more exotic launches and hyperparameter overrides. Here is a brief guide to how to set the parameters. Look at the table at the very end of the [Chinchilla paper](https://arxiv.org/abs/2203.15556) to get a sense of how the Transformer parameters (dim, n_layers, n_heads) grow or shrink together. Extrapolate/interpolate this pattern to get bigger or smaller transformers. Set the max context length however you wish, depending on the problem: this should be the max number of tokens that matter to predict the next token. E.g. Llama 2 uses 2048. Next, you want the _total_ batch size per update (printed by the script as "tokens per iteration will be:") to be somewhere around 100K tokens for medium-sized applications. For tiny applications it could be lower, for large training (e.g. GPTs/LLamas) it is usually ~0.5M, or even more. You get there by first maxing out the batch_size to whatever your system allows (e.g. mine was 16 in a recent run because after that my GPU runs out of memory), and then you want to increase gradient_accumulation_steps to be as high as necessary to reach the total batch size of ~100K. Finally, you want to tune your learning_rate (LR). You want this to be as high as your training allows. Very small networks can get away with a large LR (e.g. 1e-3 or even higher). Large networks need lower LRs. 3e-4 is a safe choice in most medium-sized applications, but can be too low for small networks, so try to increase it! Finally, max_iters is the length of training. Play with different settings. I mostly only ever tune these parameters and leave most of the others unchanged. Here is an example of how I trained the 110M model, which I don't think is anywhere near optimal, but looked sensible to me: dim 768, n_layers 12, n_heads 12 (so size of each head is 768 / 12 = 64 channels), seq len of 1024, batch size 16 (this is the most that fit my A100 40GB GPU), gradient_accumulation_steps = 8 was needed to get total tokens batch size to be 16 batch size * 1024 tokens in sequence * 8 grad_accum = 131,072 tokens per update. Good. Learning rate 4e-4 (probably a little too low). max_iters 200K (probably a bit too high). Dropout 0.1, as that usually helps a bit at medium size. That was it. I ran using Distributed Data Parallel (DDP) on 4 GPUs on my cloud machine, training took ~day or so.

Totally understand if you want to skip model training, for simple demo just download one of the pretrained models (see [models](#models) section), e.g.:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Once we have the model.bin file, we can inference in C. Compile the C code first:

```bash
make run
```

You can now run it simply as

```bash
./run stories15M.bin
```

Watch the tokens stream by, fun! We can also run the PyTorch inference script for a comparison. Download one of the models again from huggingface hub and point the `sample.py` script at it:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt -P out15M
python sample.py --checkpoint=out15M/stories15M.pt
```

Which gives the same results.

## custom tokenizers

In everything above, we've assumed the custom Lllama 2 tokenizer with 32,000 tokens. However, in many boutique LLMs, using vocabulary this big might be an overkill. If you have a small application you have in mind, you might be much better off training your own tokenizers. This can make everything nicer - with smaller vocabs your model has fewer parameters (because the token embedding table is a lot smaller), the inference is faster (because there are fewer tokens to predict), and your average sequence length per example could also get smaller (because the compression is a lot more efficient on your data). So let's see how we train a custom tokenizer.

By default, to pretokenize the tinystories dataset we had to run, in order:

```
python tinystories.py download
python tinystories.py pretokenize
```

The `pretokenize` stage here loads the Llama 2 tokenizer (vocab size 32,000) and uses it to convert the downloaded text into integers, and saves that to file. We now change this as follows, to train an example 4096-token tokenizer:

```
python tinystories.py download
python tinystories.py train_vocab --vocab_size=4096
python tinystories.py pretokenize --vocab_size=4096
```

The `train_vocab` stage will call the `sentencepiece` library to train the tokenizer, storing it in a new file `data/tok4096.model`. I tried to reproduce as well as I could the settings that (I think) Meta used to train their vocabulary. This uses the Byte Pair Encoding algorithm that starts out with raw utf8 byte sequences of the text data and then iteratively merges the most common consecutive pairs of tokens to form the vocabulary. Inspect the `tinystories.py` file - the custom tokenizers are stored in a special directory structure indexed by the vocab size.

A quick note of interest is that vocab size of 4096 trained specifically on tinystories creates integer sequences with about the same sequence length per example as the default Llama 2 tokenizer of 32000 tokens! This means that our custom, tailored tokenizer is a lot better adapted to our specific text, and can compress it very effectively. So our trained models are smaller and faster.

Now that we have pretokenized the dataset with our custom tokenizer, we can train the model. The training script `train.py` doesn't care about the exact tokens, it only cares about the vocabulary size so it can correctly initialize the model. So when training your model, make sure to pass in

```
python train.py --vocab_source=custom --vocab_size=4096
```

(The defaults are `llama2` and `32000` respectively, which indicates the default Llama 2 tokenizer). This trains the model. Finally we are ready to run inference with our `run.c` script. For that we need two things. Number one, we have to export our tokenizer in the `.bin` format, do that with:

```
python tokenizer.py --tokenizer-model=data/tok4096.model
```

This writes the tokenizer to `data/tok4096.bin`. Now we can run inference, pointing it to this tokenizer using the `-z` flag:

```
./run out/model.bin -z data/tok4096.bin
```

This should print the samples. If you leave out the `-z` flag, it will use the default Llama 2 tokenizer, which would generate a good sequence of integers, but they would get translated using a different vocabulary to text, so it would look like gibberish.

## performance

There are many ways to potentially speed up this code depending on your system. Have a look at the [Makefile](Makefile), which contains a lot of notes. The `make run` command currently uses the `-O3` optimization by default, i.e.:

```bash
gcc -O3 -o run run.c -lm
```

-O3 includes optimizations that are expensive in terms of compile time and memory usage. Including vectorization, loop unrolling, and predicting branches.

To get a much better performance, try to compile with `make runfast`. This turns on the `-Ofast` flag, which includes additional optimizations that may break compliance with the C/IEEE specifications, in addition to `-O3`. See [the GCC docs](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) for more information.

Try `-march=native` to compile the program to use the architecture of the machine you're compiling on rather than a more generic CPU. This may enable additional optimizations and hardware-specific tuning such as improved vector instructions/width.

The fastest throughput I saw so far on my MacBook Air (M1) so far is with `make runfast`.

You can also experiment with replacing `gcc` with `clang`.

If compiling with gcc, try experimenting with `-funroll-all-loops`, see PR [#183](https://github.com/karpathy/llama2.c/pull/183)

**OpenMP**. Big improvements can also be achieved by compiling with OpenMP, which "activates" the `#pragma omp parallel for` inside the matmul and attention, allowing the work in the loops to be split up over multiple processors.
You'll need to install the OpenMP library and the clang compiler first (e.g. `apt install clang libomp-dev` on ubuntu). Then you can compile with `make runomp`, which does:

```bash
clang -Ofast -fopenmp -march=native run.c  -lm  -o run
```

When you run inference make sure to use OpenMP flags to set the number of threads, e.g.:

```bash
OMP_NUM_THREADS=4 ./run out/model.bin
```

Depending on your system resources you may want to tweak these hyperparameters and use more threads. But more is not always better, usually this is a bit U shaped. In particular, if your CPU has SMT (multithreading), try setting the number of threads to the number of physical cores rather than logical cores. The performance difference can be large due to cache thrashing and communication overhead. The PyTorch documentation [CPU specific optimizations
](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#cpu-specific-optimizations) has some good information that applies here too.

## platforms

On **Windows**, use `build_msvc.bat` in a Visual Studio Command Prompt to build with msvc, or you can use `make win64` to use mingw compiler toolchain from linux or windows to build the windows target. MSVC build will automatically use openmp and max threads appropriate for your CPU unless you set `OMP_NUM_THREADS` env.

On **Centos 7**, **Amazon Linux 2018** use `rungnu` Makefile target: `make rungnu` or `make runompgnu` to use openmp.

On **Mac**, use clang from brew for openmp build. Install clang as `brew install llvm` and use the installed clang binary to compile with openmp: `make runomp CC=/opt/homebrew/opt/llvm/bin/clang`

## tests

You can run tests simply with pytest:

```bash
$ pip install pytest
$ pytest
```

This will currently invoke two tests inside `test_all.py`, which forward the model in both C and Python for 200 steps and check the output against a known good expected output. The tests currently run in only a few seconds, but will have to download and cache the stories260K models in a temporary `test` directory (only ~2MB download).

There are also some tests in C, in the file [test.c](test.c). You can run these with `make testcc`, or to see more stuff printed:

```
make testcc VERBOSITY=1
```

Call for help: help add more tests.

## ack

I trained the llama2.c storyteller models on a 4X A100 40GB box graciously provided by the excellent [Lambda labs](https://lambdalabs.com/service/gpu-cloud), thank you.

## discord

Figured it's possible to reuse my existing discord channel (that I use for my [zero to hero youtube series](https://karpathy.ai/zero-to-hero.html)), see #llama2c channel on [discord](https://discord.gg/3zy8kqD9Cp), for any quick questions, related discussions, etc.

## contributing

A few words on this repo and the kinds of PRs that are likely to be accepted. What is the goal of this repo? Basically I think there will be a lot of interest in training or finetuning custom micro-LLMs (think ~100M - ~1B params, but let's say up to ~10B params) across a large diversity of applications, and deploying them in edge-adjacent environments (think MCUs, phones, web browsers, laptops, etc.). I'd like this repo to be the simplest, smallest, most hackable repo to support this workflow, both training and inference. In particular, this repo is not a complex framework with a 1000 knobs controlling inscrutible code across a nested directory structure of hundreds of files. Instead, I expect most applications will wish to create a fork of this repo and hack it to their specific needs and deployment platforms.

People who care about deployment efficiency above all else should look at [llama.cpp](https://github.com/ggerganov/llama.cpp). This repo still cares about efficiency, but not at the cost of simplicity, readability or portability. Basically, I expect that a lot of people come to this repo because the training code is 2 readable .py files and the inference code is 500 lines of C. So I'd like this to continue to be a kind of simplest "reference implementation" that can be easily hacked in a separate fork into whatever downstream application people are excited about. It shouldn't be full-featured. It shouldn't take 100 different options or settings. It shouldn't be the most efficient. A few examples:

- someone re-ordered two loops to improve data locality for a small efficieny win => instant merge.
- someone added the one line "pragma omp parallel for", which allows you to compile with OpenMP and dramatically speed up the code, or acts as just a comment if you don't compile it that way => instant merge.
- bug fixes and touchups etc. => happy to merge

A few examples of PRs are that are not an excellent fit:

- adding more than several #ifdefs all over the place in code. If they are localized / few, might be okay.
- adding a lot of code that is very specific to some specific platform (e.g. MCUs, or some special version of linux or processor). These may be a better fit for forks of the project, and I am very happy to maintain a list of these forks in section below.
- adding hundreds of lines of code to run.c that are only active in specific scenarios or platforms.

If your candidate PRs have elements of these it doesn't mean they won't get merged, it just means they will make it into the gray territory. TLDR: I am eager to merge any mostly small, mostly localized, broadly applicable, clean changes that improve the efficiency and portability of the repo, while keep its hackability and readability. I appreciate all PRs seeking to help me improve the project, thank you! <3.

## notable forks

- Rust
  - [llama2.rs](https://github.com/gaxler/llama2.rs) by @[gaxler](https://github.com/gaxler): a Rust port of this project
  - [llama2.rs](https://github.com/leo-du/llama2.rs) by @[leo-du](https://github.com/leo-du): A Rust port of this project
  - [llama2-rs](https://github.com/danielgrittner/llama2-rs) by @[danielgrittner](https://github.com/danielgrittner): a Rust port of this project
  - [llama2.rs](https://github.com/lintian06/llama2.rs) by @[lintian06](https://github.com/lintian06): A Rust port of this project
  - [pecca.rs](https://github.com/rahoua/pecca-rs) by @[rahoua](https://github.com/rahoua): A Rust port leveraging [ndarray](https://github.com/rust-ndarray/ndarray), supports BLAS.
  - [llama2.rs](https://github.com/flaneur2020/llama2.rs) by @[flaneur2020](https://github.com/flaneur2020): A Rust port of this project.
- Go
  - [go-llama2](https://github.com/tmc/go-llama2) by @[tmc](https://github.com/tmc): a Go port of this project
  - [llama2.go](https://github.com/nikolaydubina/llama2.go) by @[nikolaydubina](https://github.com/nikolaydubina): a Go port of this project
  - [llama2.go](https://github.com/haormj/llama2.go) by @[haormj](https://github.com/haormj): a Go port of this project
  - [llama2.go](https://github.com/saracen/llama2.go) by @[saracen](https://github.com/saracen): a Go port of this project
- Android
  - [llama2.c-android](https://github.com/Manuel030/llama2.c-android): by @[Manuel030](https://github.com/Manuel030): adds Android binaries of this project
  - [llama2.c-android-wrapper](https://github.com/celikin/llama2.c-android-wrapper): by @[celikin](https://github.com/celikin): added JNI wrapper, PoC
- C++
  - [llama2.cpp](https://github.com/leloykun/llama2.cpp) by @[leloykun](https://github.com/leloykun): a C++ port of this project
- JavaScript
  - [llama2.js](https://github.com/epicure/llama2.js) by @[epicure](https://github.com/epicure): a JavaScript port of this project
  - [llamajs](https://github.com/agershun/llamajs) by @[agershun](https://github.com/agershun): a JavaScript port of this project
  - [llama2.ts](https://github.com/wizzard0/llama2.ts) by @[oleksandr_now](https://twitter.com/oleksandr_now): a TypeScript port of this project. Full Llama2-7B capable.
  - [llama2.c-emscripten](https://github.com/gohai/llama2.c-emscripten) by @[gohai](https://github.com/gohai): Emscripten (JavaScript) port, based on @ggerganov's initial prototype
- Zig
  - [llama2.zig](https://github.com/cgbur/llama2.zig) by @[cgbur](https://github.com/cgbur): A Zig port of this project
  - [llama2.zig](https://github.com/vodkaslime/llama2.zig) by @[vodkaslime](https://github.com/vodkaslime): a Zig port of this project
  - [llama2.zig](https://github.com/clebert/llama2.zig) by @[clebert](https://github.com/clebert): a Zig port of this project
- Julia
  - [llama2.jl](https://github.com/juvi21/llama2.jl) by @[juvi21](https://github.com/juvi21): a Julia port of this project
- Scala
  - [llama2.scala](https://github.com/jrudolph/llama2.scala) by @[jrudolph](https://github.com/jrudolph): a Scala port of this project
- Java
  - [llama2.java](https://github.com/mukel/llama2.java) by @[mukel](https://github.com/mukel): a Java port of this project
- Kotlin
  - [llama2.kt](https://github.com/madroidmaq/llama2.kt) by @[madroidmaq](https://github.com/madroidmaq): a Kotlin port of this project
- Python
  - [llama2.py](https://github.com/tairov/llama2.py) by @[tairov](https://github.com/tairov): a simple one file pure Python port of this project with zero dependencies
- C#
  - [llama2.cs](https://github.com/trrahul/llama2.cs) by @[trrahul](https://github.com/trrahul): a C# port of this project
- Dart
  - [llama2.dart](https://github.com/yiminghan/llama2.dart) by @[yiminghan](https://github.com/yiminghan/llama2.dart): one-file dart port of this project, works with Flutter!
- Web
  - [llama2c-web](https://github.com/dmarcos/llama2.c-web) by @[dmarcos](https://github.com/dmarcos): Super simple way to build unmodified llama2.c to WASM and run it in the browser. [Demo](https://diegomarcos.com/llama2.c-web/)
- WebAssembly
  - [icpp-llm](https://github.com/icppWorld/icpp-llm): LLMs for the Internet Computer
- Fortran
  - [llama2.f90](https://github.com/rbitr/llama2.f90): a Fortran port of this project
- Mojo
  - [llama2.🔥](https://github.com/tairov/llama2.mojo) by @[tairov](https://github.com/tairov): pure Mojo port of this project
- OCaml
  - [llama2.ml](https://github.com/jackpeck/llama2.ml) by @[jackpeck](https://github.com/jackpeck): an OCaml port of this project
- [llama2.c - Llama 2 Everywhere](https://github.com/trholding/llama2.c) by @[trholding](https://github.com/trholding): Standalone, Bootable & Portable Binary Llama 2
- [llama2.c-zh - Bilingual Chinese and English](https://github.com/chenyangMl/llama2.c-zh) by @[chenyangMl](https://github.com/chenyangMl): Expand tokenizer to support training and inference in both Chinese and English

## unsorted todos

- add support in run.c of reading version 1+ files from export, later deprecate "version 0"
- run.cu (CUDA) investigate and merge
- add more tests inside [test.c](test.c)
- add Engine class for use in sample.py that does efficient inference in PyTorch, e.g. KV cache keeping
- make it easier to add a new dataset with not too much pain
- (LoRA) finetuning and export of Llama 2 models

## License

MIT
