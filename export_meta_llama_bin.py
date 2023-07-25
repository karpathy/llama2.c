"""
This script exports the Llama 2 weights in llama2c.bin format.

Place it into the root directory of:
https://github.com/facebookresearch/llama

And then run it similar to their other examples, via torchrun sadly:
torchrun --nproc_per_node 1 export_meta_llama_bin.py
"""
import os
import sys
import llama
import time
import json
from pathlib import Path

import struct
import numpy as np
import torch
from typing import Optional
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)


# build code ported from https://github.com/krychu/llama/ by joey00072
# Thanks krychu


def build(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    model_parallel_size: Optional[int] = None,
) -> "llama.Llama":
    device = torch.device("cpu")

    if not torch.distributed.is_initialized():
        if device == "cuda":
            torch.distributed.init_process_group("nccl")
        else:
            torch.distributed.init_process_group("gloo")

    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device == "cuda":
        torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    assert model_parallel_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"

    ckpt_path = checkpoints[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: llama.ModelArgs = llama.ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    tokenizer = llama.Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.BFloat16Tensor)
    # torch.set_default_tensor_type(torch.FloatTensor) # IDK if we should use this

    model = llama.Transformer(model_args)
    model.to(device)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")

    return llama.Llama(model, tokenizer)


def export(model, filepath="model.bin"):
    """
    Export the model weights in fp32 into .bin file to be read from C.
    Args:
        model: The Llama model to be exported.
        filepath: The filepath where the model will be saved.
    """
    # Function to serialize tensor data to float32
    def serialize(t):
        d = t.detach().cpu().view(-1).numpy().astype(np.float32)
        b = struct.pack(f"{len(d)}f", *d)
        return b

    # Open the file in write mode
    with open(filepath, "wb") as f:
        # first write out the header
        hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
        p = model.params
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack(
            "iiiiiii",
            p.dim,
            hidden_dim,
            p.n_layers,
            p.n_heads,
            n_kv_heads,
            -p.vocab_size,
            p.max_seq_len,
        )
        f.write(header)

        # Write out the embedding weights
        print("Writing token embeddings...")
        f.write(serialize(model.tok_embeddings.weight))

        # Now all the layers
        layer_weights = [
            "attention_norm",
            "attention.wq",
            "attention.wk",
            "attention.wv",
            "attention.wo",
            "ffn_norm",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
        ]
        for weight in layer_weights:
            for i, layer in enumerate(model.layers):
                print(f"Writing {weight} layer {i}...")
                f.write(serialize(getattr(layer, weight).weight))

        # final rmsnorm and freqs_cis
        print("Writing final rmsnorm, classifier and freq_cis...")
        f.write(serialize(model.norm.weight))
        f.write(serialize(model.freqs_cis.real[: p.max_seq_len]))
        f.write(serialize(model.freqs_cis.imag[: p.max_seq_len]))

        # finally write the output weights
        f.write(serialize(model.output.weight))

        print(f"Wrote {filepath}")


if __name__ == "__main__":
    # Initialize Llama as normal
    generator = build(
        ckpt_dir="llama-2-7b",
        tokenizer_path="tokenizer.model",
        max_seq_len=4096,
        max_batch_size=1,
    )
    export(generator.model, "llama2_7b.bin")
