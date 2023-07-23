"""
Run simply with
$ pytest
"""
import os
import pytest # pip install pytest
import subprocess

import torch
from model import ModelArgs, Transformer

def test_argmax_inference():
    """
    Only the simplest test for now: run inference with temperature 0 
    (for determinism) in both C and PyTorch, and see that the sampled tokens 
    are the same.
    """
    test_ckpt_dir = "out" # TODO create a dummy test checkpoint for this?

    # run C version
    model_path = os.path.join(test_ckpt_dir, "model.bin")
    command = ["./run", model_path, "0.0"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    c_tokens = []
    for line in proc.stdout:
        token = int(line.decode('utf-8').strip())
        c_tokens.append(token)
    proc.wait()
    #print(c_tokens)

    # run PyTorch version
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(test_ckpt_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = ModelArgs(**checkpoint['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    x = torch.tensor([[1]], dtype=torch.long, device=device) # 1 is BOS
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=gptconf.max_seq_len, temperature=0.0)
    pt_tokens = y[0].tolist()
    pt_tokens = pt_tokens[1:] # remove BOS
    #print(pt_tokens)

    # compare
    assert c_tokens == pt_tokens
