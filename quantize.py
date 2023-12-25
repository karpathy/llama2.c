import os
import pickle
import time
from contextlib import nullcontext
import torch
import numpy as np
from numpy import inf
from spmodel import ModelArgs, Transformer
from tokenizer import Tokenizer
from export import model_export

from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
checkpoint = 'out440k_shifted_3x_25/ckpt.pt'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 4 # number of samples to draw
max_new_tokens = 128 # number of tokens generated in each sample
temperature = 0.9 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 32 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed = int(time.time())
device = 'cpu'
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float16"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

f = lambda w: torch.bincount(torch.flatten(torch.round( w/torch.max(torch.abs(w))*128).to(torch.int32)+128), minlength=257)
bins = None
p = lambda w, i: print("W", i,  torch.std_mean(w), torch.max(torch.abs(w)))
for layer in model.layers:
    if bins == None:
        bins = f(layer.feed_forward.w1.weight)
    else:
        bins += f(layer.feed_forward.w1.weight)
    bins += f(layer.feed_forward.w2.weight)
    bins += f(layer.feed_forward.w3.weight)
    
    p(layer.feed_forward.w1.weight, 1)
    p(layer.feed_forward.w2.weight, 2)
    p(layer.feed_forward.w3.weight, 3)

def h(x):
  bits = torch.log2(x)
  bits[bits == -inf] = 0
  return bits*-x

dist = bins/torch.sum(bins)

'''
print(bins)

print(dist)

print(np.round(h(dist), 2))

print(torch.sum(h(dist)))

print(torch.sum(h(dist))/8)

print(list(model.children()))
'''

model_export(model, 'out440k_shifted_3x_25/model_qint80.bin', version=2)