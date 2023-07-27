"""
This script exports the Llama 2 weights in llama2c.bin format.
"""
import os
import sys
import struct
from pathlib import Path
import json

import torch

from model import precompute_freqs_cis


def export(p, state_dict, filepath='model.bin'):
    """export the model weights in fp32 into .bin file to be read from C"""
    f = open(filepath, 'wb')

    def serialize(key):
        print(f"writing {key}...")
        t = state_dict[key].contiguous().view(-1).type(torch.float32).numpy()
        f.write(memoryview(t))
        del state_dict[key]

    # first write out the header
    hidden_dim = state_dict['layers.0.feed_forward.w1.weight'].shape[0]
    p['vocab_size'] = 32000
    p['max_seq_len'] = 2048

    n_kv_heads = p.get('n_kv_heads') or p['n_heads']
    header = struct.pack(
        'iiiiiii',
        p['dim'], hidden_dim, p['n_layers'], p['n_heads'],
        n_kv_heads, -p['vocab_size'], p['max_seq_len']
    )
    # NOTE ABOVE: -ve vocab_size is indicating that the classifier weights are present
    # in the checkpoint and should be loaded.
    f.write(header)

    # next write out the embedding weights
    print("writing tok_embeddings...")
    serialize('tok_embeddings.weight')

    # now all the layers
    # attention weights
    for i in range(p['n_layers']): serialize(f'layers.{i}.attention_norm.weight')
    for i in range(p['n_layers']): serialize(f'layers.{i}.attention.wq.weight')
    for i in range(p['n_layers']): serialize(f'layers.{i}.attention.wk.weight')
    for i in range(p['n_layers']): serialize(f'layers.{i}.attention.wv.weight')
    for i in range(p['n_layers']): serialize(f'layers.{i}.attention.wo.weight')
    # ffn weights
    for i in range(p['n_layers']): serialize(f'layers.{i}.ffn_norm.weight')
    for i in range(p['n_layers']): serialize(f'layers.{i}.feed_forward.w1.weight')
    for i in range(p['n_layers']): serialize(f'layers.{i}.feed_forward.w2.weight')
    for i in range(p['n_layers']): serialize(f'layers.{i}.feed_forward.w3.weight')

    # final rmsnorm
    serialize('norm.weight')
    # freqs_cis
    freqs_cis = precompute_freqs_cis(p['dim'] // p['n_heads'], p['max_seq_len'] * 2)
    state_dict['freqs_cis.real'] = freqs_cis.real[:p['max_seq_len']]
    state_dict['freqs_cis.imag'] = freqs_cis.imag[:p['max_seq_len']]
    serialize('freqs_cis.real')
    serialize('freqs_cis.imag')

    # finally write the output weights
    serialize('output.weight')

    f.close()
    print(f"wrote {filepath}")


def concat_weights(models):
    state_dict = {}
    for name in list(models[0]):
        tensors = [model[name] for model in models]
        if len(tensors) == 1 or len(tensors[0].shape) == 1:
            state_dict[name] = tensors[0]
            continue
        is_axis_1 = (
            name.startswith('tok_embeddings.')
            or name.endswith('.attention.wo.weight')
            or name.endswith('.feed_forward.w2.weight')
        )
        axis = 1 if is_axis_1 else 0
        state_dict[name] = torch.cat(tensors, dim=axis)
        for model in models:
            del model[name]
    return state_dict


def load_and_export(model_path, output_path):
    params_path = os.path.join(model_path, 'params.json')
    with open(params_path) as f:
        params = json.load(f)
        print(params)

    model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
    models = [torch.load(p, map_location='cpu') for p in model_paths]
    state_dict = concat_weights(models)
    del models
    export(params, state_dict, output_path)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('[Llama model folder path] [output path]')
        exit()

    model_path = sys.argv[1]
    output_path = sys.argv[2]
    load_and_export(model_path, output_path)
