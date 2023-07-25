"""
This script exports the Llama 2 weights in llama2c.bin format.

Place it into the root directory of:
https://github.com/facebookresearch/llama

And then run it similar to their other examples, via torchrun sadly:
torchrun --nproc_per_node 1 export_meta_llama_bin.py
"""

from llama import Llama
import struct
import numpy as np


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
    generator = Llama.build(
        ckpt_dir="llama-2-7b",
        tokenizer_path="tokenizer.model",
        max_seq_len=4096,
        max_batch_size=1,
    )
    export(generator.model, "llama2_7b.bin")
