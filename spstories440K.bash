#!/bin/bash

python train.py \
    --out_dir="out440k_shifted_3x_25" \
    --batch_size=512 \
    --max_seq_len=64 \
    --gradient_accumulation_steps=1 \
    --vocab_source="custom" \
    --vocab_size=1024 \
    --dim=64 \
    --n_layers=4 \
    --n_heads=4 \
    --n_kv_heads=4 \
    --multiple_of=32 \
    --learning_rate=1e-3 \
    --dropout=0.05 \
    --weight_decay=0.01 \
    --max_iters=100000 \
    --beta2=0.99 \
    --warmup_iters=1000 \
    --eval_interval=2000 \
    --eval_iters=100 \
    --compile=True \
    --device="cuda:0" \
    --dtype="float16" \
    --sparse=True