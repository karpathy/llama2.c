#!/bin/bash

python train.py \
    --out_dir="out2M" \
    --batch_size=128 \
    --max_seq_len=128 \
    --gradient_accumulation_steps=1 \
    --vocab_source="custom" \
    --vocab_size=4096 \
    --dim=128 \
    --n_layers=6 \
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
    --device="cuda:2" \
    --dtype="float32"