"""
This script exports the Llama 2 weights in llama2c.bin format.

Place it into the root directory of:
https://github.com/facebookresearch/llama

And then run it similar to their other examples, via torchrun sadly:
torchrun --nproc_per_node 1 export_meta_llama_bin.py
"""

from llama import Llama

# -----------------------------------------------------------------------------
def export(self, filepath='model.bin'):
    """export the model weights in fp32 into .bin file to be read from C"""

    f = open(filepath, 'wb')
    import struct
    import numpy as np

    def serialize(t):
        d = t.detach().cpu().view(-1).numpy().astype(np.float32)
        b = struct.pack(f'{len(d)}f', *d)
        f.write(b)

    # first write out the header
    hidden_dim = self.layers[0].feed_forward.w1.weight.shape[0]
    p = self.params
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads, 
                                    n_kv_heads, -p.vocab_size, p.max_seq_len)
    # NOTE ABOVE: -ve vocab_size is indicating that the classifier weights are present
    # in the checkpoint and should be loaded.
    f.write(header)

    # next write out the embedding weights
    print("writing tok_embeddings...")
    serialize(self.tok_embeddings.weight)
    
    # now all the layers
    # attention weights
    for i, layer in enumerate(self.layers):
        print(f"writing attention_norm layer {i}...")
        serialize(layer.attention_norm.weight)
    for i, layer in enumerate(self.layers):
        print(f"writing attention.wq layer {i}...")
        serialize(layer.attention.wq.weight)
    for i, layer in enumerate(self.layers):
        print(f"writing attention.wk layer {i}...")
        serialize(layer.attention.wk.weight)
    for i, layer in enumerate(self.layers):
        print(f"writing attention.wv layer {i}...")
        serialize(layer.attention.wv.weight)
    for i, layer in enumerate(self.layers):
        print(f"writing attention.wo layer {i}...")
        serialize(layer.attention.wo.weight)
    # ffn weights
    for i, layer in enumerate(self.layers):
        print(f"writing ffn_norm layer {i}...")
        serialize(layer.ffn_norm.weight)
    for i, layer in enumerate(self.layers):
        print(f"writing feed_forward.w1 layer {i}...")
        serialize(layer.feed_forward.w1.weight)
    for i, layer in enumerate(self.layers):
        print(f"writing feed_forward.w2 layer {i}...")
        serialize(layer.feed_forward.w2.weight)
    for i, layer in enumerate(self.layers):
        print(f"writing feed_forward.w3 layer {i}...")
        serialize(layer.feed_forward.w3.weight)
    # final rmsnorm
    print("writing final rmsnorm, classifier and freq_cis...")
    serialize(self.norm.weight)
    # freqs_cis
    serialize(self.freqs_cis.real[:p.max_seq_len])
    serialize(self.freqs_cis.imag[:p.max_seq_len])
    # finally write the output weights
    serialize(self.output.weight)

    # write to binary file
    f.close()
    print(f"wrote {filepath}")
# -----------------------------------------------------------------------------

# init Llama as normal
generator = Llama.build(
    ckpt_dir="llama-2-7b",
    tokenizer_path="tokenizer.model",
    max_seq_len=4096,
    max_batch_size=1,
)
export(generator.model, "llama2_7b.bin")
