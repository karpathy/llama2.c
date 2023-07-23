import argparse
import importlib
from pprint import pprint


def filter_dict(d):
    return {
        _k: _v
        for _k, _v in d.items()
        if not _k.startswith("_") and isinstance(_v, (int, float, bool, str))
    }


def make_header(text, n=35):
    n = max(n, len(text))
    print()
    print('-' * n)
    print(text)
    print('-' * n)


def update_config(g, verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='*', default=[], type=str)
    config_keys = filter_dict(g)
    for k, default in config_keys.items():
        _type = type(default)
        parser.add_argument(f'--{k}', metavar="", type=_type, help=f"{_type.__name__} (default: {default})")
    args = parser.parse_args()

    # Load the variables defined in the extra config files
    for extra_config_file in args.config_files:
        extra_config = filter_dict(importlib.import_module(extra_config_file.removesuffix('.py').replace('/', '.')).__dict__)
        if verbose:
            make_header(f'Loaded from extra config file {extra_config_file}:')
            pprint(extra_config)
        g.update(extra_config)

    # Update variables that have been set via CLI
    cli_params = {k: v for k, v in args.__dict__.items() if v is not None}
    if verbose:
        make_header('Loaded from CLI params:')
        pprint(cli_params)
    g.update(cli_params)
    config = {k: g[k] for k in config_keys}  # will be useful for logging
    if verbose:
        make_header('Full config:')
        pprint(config)
    return config
