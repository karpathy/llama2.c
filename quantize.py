import os
import pickle
import time
from contextlib import nullcontext
import torch
import matplotlib.pyplot as plt
import numpy as np
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

print("From w1-3 alone: ")
print(bins)

print(dist)

print(np.round(h(dist), 2))

print(torch.sum(h(dist)))

print(torch.sum(h(dist))/8)

print(list(model.children()))

print("From export: ")
bins = torch.tensor([    0,    13,     0,     1,     1,     1,     2,     3,     2,     2,
            2,     1,     0,     4,     2,     3,     4,     5,     1,     6,
            7,   195,    24,    15,    12,     8,     7,     7,     7,    11,
            4,     8,     9,     9,     8,    14,    12,    27,    44,   205,
           25,    53,   226,    29,    25,    33,    24,    20,    28,    27,
           23,    36,    32,    29,    48,    32,    44,    66,    63,   280,
          104,    60,   302,   126,    99,   116,   142,   141,   320,   419,
          339,   168,   241,   294,   217,   218,   267,   253,   265,   284,
          293,   315,   575,   443,   489,  1099,   556,   756,   632,   955,
         1234,   819,   777,   872,  1078,  1143,  1066,  1287,  1209,  1565,
         1351,  1435,  1745,  1870,  2029,  2106,  1996,  2448,  2558,  2491,
         2365,  2791,  2809,  3141,  3123,  3277,  3480,  3776,  3926,  4102,
         4316,  4612,  4892,  5686,  6240,  6820,  8006, 11667, 23788, 11425,
         7767,  6964,  5860,  5233,  4976,  4712,  4394,  4247,  3954,  3803,
         3719,  3556,  3321,  3232,  3198,  3118,  2903,  2595,  2258,  2096,
         1947,  1839,  1793,  1642,  1553,  1440,  1386,  1236,  1187,  1168,
         1113,  1196,   925,  1041,   746,   708,   693,   657,   578,   733,
          544,   519,   732,   399,   396,   558,   356,   338,   499,   307,
          614,   455,   274,   547,   176,   189,   181,   299,   135,    99,
           98,    88,    88,    76,    76,    72,    84,   100,   105,   233,
          200,    37,    40,    35,    34,    31,    30,    35,    27,    29,
           16,    18,    13,    18,    14,    13,    11,    10,    18,     9,
            6,     8,     6,     6,     5,     4,     4,     9,     6,     7,
            2,     6,     1,     3,     5,     3,     0,     5,     4,     0,
            1,     3,     3,     0,     2,     1,     5,     0,     0,     1,
            2,     0,     0,     2,     1,    17])

dist = bins/torch.sum(bins)

print(bins)

print(dist)

print(np.round(h(dist), 2))

print(torch.sum(h(dist)))

print(torch.sum(h(dist))/8)

plt.plot(range(-128, 128), dist)

bins = torch.Tensor([8, 8, 6, 4, 1, 11, 5, 1, 1, 3, 7, 7, 7, 8, 9, 8, 11, 6, 1, 11, 4, 7, 6, 8, 15, 12, 6, 9, 8, 6, 5, 8, 6, 9, 14, 10, 18, 8, 6, 10, 14, 14, 15, 14, 13, 19, 12, 14, 21, 18, 18, 18, 24, 20, 17, 22, 21, 18, 31, 27, 27, 26, 35, 26, 17, 36, 48, 32, 38, 43, 46, 49, 46, 32, 48, 52, 47, 50, 72, 63, 63, 63, 64, 72, 75, 64, 78, 72, 78, 88, 90, 77, 88, 95, 120, 110, 112, 104, 129, 125, 126, 127, 123, 160, 136, 160, 192, 177, 188, 188, 222, 213, 267, 294, 351, 323, 335, 441, 480, 604, 575, 697, 812, 960, 1081, 1249, 1398, 1527, 1549, 1249, 1137, 994, 800, 679, 606, 504, 491, 418, 412, 361, 340, 264, 270, 248, 214, 210, 200, 214, 169, 176, 167, 147, 173, 144, 158, 165, 117, 127, 114, 101, 119, 131, 107, 111, 89, 79, 93, 100, 83, 94, 83, 72, 66, 87, 71, 52, 62, 50, 69, 60, 55, 57, 42, 60, 52, 43, 51, 36, 34, 28, 28, 28, 23, 30, 26, 33, 30, 27, 21, 23, 16, 21, 23, 14, 23, 20, 21, 18, 20, 20, 17, 12, 13, 14, 10, 16, 16, 12, 9, 16, 8, 9, 14, 11, 12, 11, 10, 4, 7, 6, 11, 9, 2, 4, 4, 3, 8, 5, 6, 3, 11, 4, 4, 2, 3, 1, 2, 1, 1, 7, 6, 5, 3, 2, 102, 0])
dist = bins/torch.sum(bins)

print(bins)

print(dist)

print(np.round(h(dist), 2))

print(torch.sum(h(dist)))

print(torch.sum(h(dist))/8)

plt.plot(range(-128, 128), dist)
plt.ylabel('Probability')
plt.xlabel('Data')
plt.savefig("WeightCacheDist.png")

#model_export(model, 'out440k_shifted_3x_25/model_qint80.bin', version=2)
model_export(model, 'out440k_shifted_3x_25/model_mcu.bin', version=3)