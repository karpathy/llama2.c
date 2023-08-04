#!/usr/bin/env python
"""Saves the model as a TorchScript.

The resulting file can be loaded in C++ code and then used for training or
inference with:
    #include <torch/script.h>
    torch::jit::Module module = torch::jit::load("model.pt")

Note that the serialized model includes the initial parameters and with the default
ModelArgs the file is 59M and gzips down to 55M. If you want to serialize/distribute
the model parameters separately and you can zero out the parameters before saving it
and it will gzip down to 780K:
    for p in model.parameters():
        p.detach().zero_()
"""
import glob
import os
import sys
from typing import List

import torch

from model import ModelArgs, Transformer


def main() -> None:
    model = Transformer(
        ModelArgs(
            dim=288,
            n_layers=6,
            n_heads=6,
            multiple_of=32,
            dropout=0.0,
            vocab_size=32000,
        )
    )
    torch.jit.save(torch.jit.script(model), "model.pt")


if __name__ == "__main__":
    main()
