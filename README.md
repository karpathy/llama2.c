
## llama2.c

Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure C? No? Well, now you can!

<img src="assets/llama_cute.jpg" width="300" height="300">

With this code you can train the Llama 2 LLM architecture from scratch in PyTorch, then save the weights to a raw binary file, then load that into one ~simple 500-line C file ([run.c](run.c)) that inferences the model, simply in fp32 for now. On my cloud Linux devbox a dim 288 6-layer 6-head model (~15M params) inferences at ~100 tok/s in fp32, and about the same on my M1 MacBook Air. I was somewhat pleasantly surprised that one can run reasonably sized models (few ten million params) at highly interactive rates with an approach this simple.

Please note that this is just a weekend project: I took nanoGPT, tuned it to implement the Llama-2 architecture instead of GPT-2, and the meat of it was writing the C inference engine in [run.c](run.c). As such, this is not really meant to be a production-grade library right now.

Hat tip to [llama.cpp](https://github.com/ggerganov/llama.cpp) for inspiring this project. I wanted something super minimal so I chose to hard-code the llama-2 architecture, stick to fp32, and just roll one inference file of pure C with no dependencies.

## feel the magic

Let's just run a baby Llama 2 model in C. You need a model checkpoint. Download this 15M parameter model I trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (~58MB download) and place it into the default checkpoint directory `out`:

```bash
wget https://karpathy.ai/llama2c/model.bin -P out
```

(if that doesn't work try [google drive](https://drive.google.com/file/d/1aTimLdx3JktDXxcHySNrZJOOk8Vb1qBR/view?usp=share_link)). Compile and run the C code (check [howto](#howto) for faster optimization flags):

```bash
gcc -O3 -o run run.c -lm
./run out/model.bin
```

You'll notice that this just streams the raw tokens. Unless you can read those directly, you'll want to translate them into text. For now sadly we have to run this C code through a simple wrapper that does the translation (see the file, it's just 30 lines):

```bash
pip install sentencepiece
python run_wrap.py
```

You'll see text stream. On my M1 MacBook Air this runs at ~100 tokens/s, not bad for super naive fp32 single-threaded C code. Sample output:

*Once upon a time, there was a boy named Timmy. Timmy loved to play sports with his friends. He was very good at throwing and catching balls. One day, Timmy's mom gave him a new shirt to wear to a party. Timmy thought it was impressive and asked his mom to explain what a shirt could be for. "A shirt is like a special suit for a basketball game," his mom said. Timmy was happy to hear that and put on his new shirt. He felt like a soldier going to the army and shouting. From that day on, Timmy wore his new shirt every time he played sports with his friends at the party. Once upon a time, there was a little girl named Lily. She loved to play outside with her friends. One day, Lily and her friend Emma were playing with a ball. Emma threw the ball too hard and it hit Lily's face. Lily felt embarrassed and didn't want to play anymore.
Emma asked Lily what was wrong, and Lily told her about her memory. Emma told Lily that she was embarrassed because she had thrown the ball too hard. Lily felt bad
achieved tok/s: 98.746993347843922*

## howto

It should be possible to load the weights released by Meta but I haven't tried because the inference speed, even of the 7B model, would probably be not great with this baby single-threaded C program. So in this repo we focus on more narrow applications, and train the same architecture but from scratch, in this case on the TinyStories dataset for fun.

First let's download and pretokenize some source dataset, e.g. I like [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) so this is the only example currently available in this repo. But it should be very easy to add datasets, see the code.

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

Alternatively, if you want to increase the inference performance and are confident in using unsafe math optimizations, which are probably fine for this application, you can compile the code with the `-funsafe-math-optimizations` flag as shown below:

```bash
gcc -O3 -funsafe-math-optimizations -o run run.c -lm
```

You can now run it simply as

```bash
./run out/model.bin
```

But note that this only emits the SentencePiece tokens. To decode the tokens into text too, run this script through a simple wrapper:

```bash
python run_wrap.py
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

## unsorted todos

- why SentencePiece can't iteratively decode properly?
- would love to delete run_wrap.py and just directly use C code to string
- todo multiquery support? doesn't seem as useful for smaller models that run on CPU (?)
- todo support inferencing beyond max_seq_len steps, have to think through the kv cache
- why is MFU so low (~10%) on my A100 40GB for training?
- weird errors with torch.compile and wandb when using DDP
- make more better tests to decrease yolo

## License
MIT
