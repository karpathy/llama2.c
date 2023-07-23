
## llama2.c

![llama2c](assets/llama_cute.jpg)

Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure C? No? Well, now you can!

Code in this repo first lets you train the Llama 2 architecture from scratch in PyTorch, then save the weights to a raw binary file, then load that into one ~simple 500-line C file that inferences the model, simply in fp32 for now.

Of course, this is not super fast, but it's not too bad either. E.g. on my cloud Linux devbox a dim 288 6-layer 6-head model (~15M params) inferences at ~18 tok/s in fp32, and about the same on my M1 MacBook Air.

Please note that this is just a weekend project where I took nanoGPT, gutted it to implement the Llama-2 architecture (instead of GPT-2), and then wrote the C inference engine for it in `run.c`. So this is not really meant to be a production-grade library right now.

Hat tip to [llama.cpp](https://github.com/ggerganov/llama.cpp) for inspiring this project. I wanted something super minimal so I chose to hard-code the llama-2 architecture, stick to fp32, and just roll one inference file of pure C with no dependencies.

## howto

It should be possible to load the weights released by Meta but I haven't tried because the inference speed, even of the 7B model, would probably be not great with this baby single-threaded C program. So in this repo we focus on more narrow applications, and train the same architecture but from scratch, in this case on the TinyStories dataset for fun.

First let's download and pretokenize the TinyStories dataset:

```bash
python tinystories.py download
python tinystories.py pretokenize
```

Then train our model:

```bash
python train.py
```

See the train.py script for more exotic launches and hyperparameter overrides. I didn't tune the hyperparameters, I expect simple hyperparameter exploration should give better models. Totally understand if you want to skip model training, for simple demo just download my pretrained model:

```bash
wget TODOhoweasiesthmm
```

Once we have the model.bin file, we can inference in C. Compile the C code first:

```bash
gcc -o run run.c -lm
```

You can now run it simply as

```bash
./run
```

But note that this only emits the SentencePiece tokens. To decode the tokens into text too, run this script through a simple wrapper:

```bash
python run_wrap.py
```

I hope to delete this script soon though. Anyway, watch the tokens stream by, fun!

To verify correctness, we can also run the PyTorch inference script:

```bash
python sample.py
```

Which gives the same results. I'd love to find some time to create actual tests, one day maybe. For now I just manually inspected activations and verified that they match, and that the samples are identical at temperature 0. If someone wishes to help me with tests I welcome PRs.

## unsorted todos

- why SentencePiece can't iteratively decode properly?
- would love to delete run_wrap.py and just directly use C code to string, help welcome
- todo multiquery support? doesn't seem as useful for smaller models that run on CPU
- todo support inferencing beyond max_seq_len steps, have to think through the kv cache
- why is MFU so low (~20%) on my A100 40GB for training?
- weird errors with torch.compile and wandb when using DDP
- make tests to decrease yolo

## License
MIT
