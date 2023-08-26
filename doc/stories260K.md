# stories260K

[Stories260K huggginface link](https://huggingface.co/karpathy/tinyllamas)

The 260K model is a tiny model used for testing, and was trained as follows:

```
python train.py \
    --out_dir="outmini" \
    --batch_size=128 \
    --max_seq_len=512 \
    --gradient_accumulation_steps=1 \
    --vocab_source="custom" \
    --vocab_size=512 \
    --dim=64 \
    --n_layers=5 \
    --n_heads=8 \
    --n_kv_heads=4 \
    --multiple_of=4 \
    --learning_rate=1e-3 \
    --dropout=0.05 \
    --weight_decay=0.01 \
    --max_iters=100000 \
    --beta2=0.99 \
    --warmup_iters=1000 \
    --eval_interval=2000 \
    --eval_iters=100 \
    --compile=True
```

You'll notice that `n_kv_heads` is 4 while `n_heads` is 8, so two heads at a time share their key,value projections, i.e. this model is 2X multiquery. You'll also notice that we're using a custom tokenizer with 512 tokens. The model trained for ~10 minutes (?) on my A100 and achieves validation loss of 1.2968.

Sampling this model at temperature 0.0 (i.e. deterministic greedy argmax sampling) gives:

```
$ ./run stories260K/stories260K.bin -z stories260K/tok512.bin -t 0.0
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you.
```

You can reproduce the same in Python by running `sample.py`:

```
$ python sample.py --checkpoint=stories260K/stories260K.pt --tokenizer=stories260K/tok512.model --temperature=0.0 --max_new_tokens=257
```

I hardcoded max tokens to be 257 manually because the `sample.py` script doesn't currently terminate on the special BOS token like the run.c script does. Sampling at 1.0 with topp of 0.9 gives a bit more reasonable samples:

```
$ ./run stories260K/stories260K.bin -z stories260K/tok512.bin -t 1.0 -p 0.9 -s 133742
Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toys and eat sandwiches. One day, Timmy's mom told him it was time to rest for a while. Timmy's friend Billy came over and took him a down.
Timmy's mom saw that Timmy was sad, but Timmy said, "I didn't understand what is it! We need to find some leafs." Timmy thought about it and took a deep breath on a spoon. He hoped it was important to be kind and continued to find its image next time.
After they finished getting, Timmy's dad came up to his house and promised to help Timmy.
```

Hey you can't expect too much from a 260K parameter model. I'm even mildly shocked we get this far :D
