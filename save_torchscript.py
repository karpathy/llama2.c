#!/usr/bin/env python
"""Saves the model as a TorchScript.

Usage examples:
    ./save_torchscript.py
    ./save_torchscript.py --dim=300
    ./save_torchscript.py --gzip_output=True --zero_params=True

The resulting file can be loaded in C++ code and then used for training or
inference with:
    #include <torch/script.h>
    torch::jit::Module module = torch::jit::load("model.pt")

Note that the serialized model includes the initial parameters and with the default
ModelArgs the file is 59M and gzips down to 55M. If you want to serialize/distribute
the model parameters separately you can zero out the parameters before saving it and
it will gzip down to 780K.
"""
import gzip
import os
import shutil
from inspect import signature

import torch

from model import ModelArgs, Transformer

# Model args config
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = n_heads
multiple_of = 32
max_seq_len = 256
dropout = 0.0
vocab_size = 32000
norm_eps = 1e-5
# Save config
model_path = "model.pt"
zero_params = False
gzip_output = False
# Allow config overrides
exec(open("configurator.py").read())


def main() -> None:
    model_args = {k: globals()[k] for k in signature(ModelArgs).parameters}
    model = Transformer(ModelArgs(**model_args))

    # If requested zero params before saving the model. This is useful in
    # conjunction with gzip_output.
    if zero_params:
        for p in model.parameters():
            p.detach().zero_()

    torch.jit.save(torch.jit.script(model), model_path)

    if gzip_output:
        with open(model_path, "rb") as f_in:
            with gzip.open(f"{model_path}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.unlink(model_path)


if __name__ == "__main__":
    main()
