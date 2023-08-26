"""
Sample from the trained model with PyTorch
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer, RMSNorm, TransformerBlock
from tokenizer import Tokenizer

from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
checkpoint = 'test/stories260K.pt'
start = "Lily was a happy girl" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "test/tok512.model" # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
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

def RMSNorm_forward(self, x):
        if kv_cache.get(id(self)) is None:
            output = self._norm(x.float()).type_as(x)
            output = output * self.weight
            kv_cache[id(self)] = output
        else:
            output = self._norm(x[..., -1:, :].float()).type_as(x)
            output = output * self.weight
            output = torch.cat((kv_cache[id(self)], output), dim=-2)
            kv_cache[id(self)] = output
        return output

RMSNorm.forward = RMSNorm_forward
# class KVC_RMSNorm(RMSNorm):
#     def forward(self, x):
#         if not kv_cache.get(id(self)):
#             output = self._norm(x.float()).type_as(x)
#             output = output * self.weight
#             kv_cache[id(self)] = output
#         else:
#             output = self._norm(x[-1:].float()).type_as(x)
#             output = output * self.weight
#             kv_cache[id(self)] = kv_cache[id(self)].cat(output, dim=0)
#         return output * self.weight

# class KVC_TransformerBlock(TransformerBlock):
#     def __init__(self, layer_id: int, args: ModelArgs):
#         super().__init__(layer_id, args)
#         self.attention_norm = KVC_RMSNorm(args.dim, eps=args.norm_eps)

#     def forward(self, x, freqs_cos, freqs_sin):
#         h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
#         out = h + self.feed_forward.forward(self.ffn_norm(h))
#         return out

kv_cache = {}
# class Engine(Transformer):

#     def forward(self, tokens: torch.Tensor) -> torch.Tensor:
#         _bsz, seqlen = tokens.shape
#         h = self.tok_embeddings(tokens)
#         freqs_cos = self.freqs_cos[:tokens.shape[1]]
#         freqs_sin = self.freqs_sin[:tokens.shape[1]]

#         for layer in self.layers:
#             h = layer(h, freqs_cos, freqs_sin)

#         h = self.norm(h)

#         logits = self.output(h[:, [-1], :])
#         self.last_loss = None

#         return logits



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

# load the tokenizer
vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
vocab_size = gptconf.vocab_size
if tokenizer:
    # a specific tokenizer is provided, use it
    tokenizer_model = tokenizer
else:
    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
enc = Tokenizer(tokenizer_model=tokenizer_model)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = enc.encode(start, bos=True, eos=False)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y[0].tolist()))
            print('---------------')
