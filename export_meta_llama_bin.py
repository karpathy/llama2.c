"""
This script exports the Llama 2 weights in llama2c.bin format.

Place it into the root directory of:
https://github.com/facebookresearch/llama

And then run:
python export_meta_llama_bin.py
"""

import json
from pathlib import Path

import torch

# -----------------------------------------------------------------------------
def export(checkpoint, params, filepath='model.bin'):
    """export the model weights in fp32 into .bin file to be read from C"""

    f = open(filepath, 'wb')
    import struct
    import numpy as np

    def serialize1(t):
        d = t.detach().cpu().view(-1).numpy()
        b = d.tobytes()
        f.write(b)

    def serialize(t):
        d = t.detach().cpu().float().numpy()
        b = d.flatten().tobytes()
        f.write(b)

    def serialize_old(t):
        d = t.detach().cpu().view(-1).numpy().astype(np.float32)
        b = struct.pack(f'{len(d)}f', *d)
        f.write(b)

    # first write out the header
    hidden_dim = checkpoint['layers.0.feed_forward.w1.weight'].shape[0] #self.layers[0].feed_forward.w1.weight.shape[0]
    p = params
    p['max_seq_len'] = 4096
    p['vocab_size'] = 32000
    n_layers = p['n_layers']
    n_kv_heads = p.get('n_kv_heads', p['n_heads'])
    # header magic version integer added for two reasons
    # 1) so that we can version the header
    # 2) so that the struct maintains strict cache alignment
    #    which is necessary so that the weights that follow the header are also cache aligned
    header_magic_version = 0x42000000
    header = struct.pack('iiiiiiii', header_magic_version, p['dim'], hidden_dim, n_layers, p['n_heads'], 
                                    n_kv_heads, -p['vocab_size'], p['max_seq_len'])
    # NOTE ABOVE: -ve vocab_size is indicating that the classifier weights are present
    # in the checkpoint and should be loaded.
    f.write(header)

    # next write out the embedding weights
    print("writing tok_embeddings...")
    serialize(checkpoint['tok_embeddings.weight'].type(torch.HalfTensor))
    
    # now all the layers
    # attention weights
    for i in range(n_layers):
        print(f"writing attention_norm layer {i}...")
        serialize(checkpoint['layers.'+str(i)+'.attention_norm.weight'].type(torch.HalfTensor))
    for i in range(n_layers):
        print(f"writing attention.wq layer {i}...")
        serialize(checkpoint['layers.'+str(i)+'.attention.wq.weight'].type(torch.HalfTensor))
    for i in range(n_layers):
        print(f"writing attention.wk layer {i}...")
        serialize(checkpoint['layers.'+str(i)+'.attention.wk.weight'].type(torch.HalfTensor))
    for i in range(n_layers):
        print(f"writing attention.wv layer {i}...")
        serialize(checkpoint['layers.'+str(i)+'.attention.wv.weight'].type(torch.HalfTensor))
    for i in range(n_layers):
        print(f"writing attention.wo layer {i}...")
        serialize(checkpoint['layers.'+str(i)+'.attention.wo.weight'].type(torch.HalfTensor))
    # ffn weights
    for i in range(n_layers):
        print(f"writing ffn_norm layer {i}...")
        serialize(checkpoint['layers.'+str(i)+'.ffn_norm.weight'].type(torch.HalfTensor))
    for i in range(n_layers):
        print(f"writing feed_forward.w1 layer {i}...")
        serialize(checkpoint['layers.'+str(i)+'.feed_forward.w1.weight'].type(torch.HalfTensor))
    for i in range(n_layers):
        print(f"writing feed_forward.w2 layer {i}...")
        serialize(checkpoint['layers.'+str(i)+'.feed_forward.w2.weight'].type(torch.HalfTensor))
    for i in range(n_layers):
        print(f"writing feed_forward.w3 layer {i}...")
        serialize(checkpoint['layers.'+str(i)+'.feed_forward.w3.weight'].type(torch.HalfTensor))


    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    freqs_cis = precompute_freqs_cis(
        p['dim'] // p['n_heads'], p['max_seq_len'] * 2
    )

    # final rmsnorm
    print("writing final rmsnorm, classifier and freq_cis...")
    serialize(checkpoint['norm.weight'].type(torch.HalfTensor))
    # freqs_cis
    serialize(freqs_cis.real[:p['max_seq_len']].type(torch.HalfTensor))
    serialize(freqs_cis.imag[:p['max_seq_len']].type(torch.HalfTensor))
    # finally write the output weights
    serialize(checkpoint['output.weight'].type(torch.HalfTensor))

    # write to binary file
    f.close()
    print(f"wrote {filepath}")
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    ckpt_dir = "llama-2-7b"
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpt_path = checkpoints[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    export(checkpoint, params, "llama2_7b.bin")
