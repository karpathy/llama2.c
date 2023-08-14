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
    hidden_dim = state_dict['model.layers.0.mlp.gate_proj.weight'].shape[0]
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
    serialize('model.embed_tokens.weight')

    # now all the layers
    # attention weights
    for i in range(p['n_layers']): serialize(f'model.layers.{i}.input_layernorm.weight')
    for i in range(p['n_layers']): serialize(f'model.layers.{i}.self_attn.q_proj.weight')
    for i in range(p['n_layers']): serialize(f'model.layers.{i}.self_attn.k_proj.weight')
    for i in range(p['n_layers']): serialize(f'model.layers.{i}.self_attn.v_proj.weight')
    for i in range(p['n_layers']): serialize(f'model.layers.{i}.self_attn.o_proj.weight')
    # ffn weights
    for i in range(p['n_layers']): serialize(f'model.layers.{i}.post_attention_layernorm.weight')
    for i in range(p['n_layers']): serialize(f'model.layers.{i}.mlp.gate_proj.weight')
    for i in range(p['n_layers']): serialize(f'model.layers.{i}.mlp.down_proj.weight')
    for i in range(p['n_layers']): serialize(f'model.layers.{i}.mlp.up_proj.weight')

    # final rmsnorm
    serialize('model.norm.weight')
    # freqs_cos, freqs_sin
    freqs_cos, freqs_sin = precompute_freqs_cis(p['dim'] // p['n_heads'], p['max_seq_len'] * 2)
    state_dict['freqs_cos'] = freqs_cos[:p['max_seq_len']]
    state_dict['freqs_sin'] = freqs_sin[:p['max_seq_len']]
    # check if this requires addtional conversion
    serialize('freqs_cos')
    serialize('freqs_sin')

    # finally write the output weights
    serialize('lm_head.weight')

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
            name.startswith('model.embed_tokens.weight')
            or name.endswith('.self_attn.o_proj.weight')
            or name.endswith('.mlp.down_proj.weight')
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
