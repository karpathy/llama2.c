"""
Download, preprocess and serve the arbitrary dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url and file extension"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(dataset_id, data_url, data_type):
    """Downloads the dataset to disk."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the dataset, unless it's already downloaded
    data_filename = os.path.join(DATA_CACHE_DIR, f"{dataset_id}.{data_type}")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files), else copy data(e.g., *.json, *.txt) into dataset_id directory.
    data_dir = os.path.join(DATA_CACHE_DIR, f"{dataset_id}")
    if not os.path.exists(data_dir):
        if data_filename.endswith(".tar.gz"):
            os.makedirs(data_dir, exist_ok=True)
            print(f"Unpacking {data_filename}...")
            os.system(f"tar -xzf {data_filename} -C {data_dir}")
        else:
            os.makedirs(data_dir, exist_ok=True)
            print(f"Copying {data_filename}...")
            os.system(f"cp {data_filename} {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping processing...")

    print("Download done.")


def process_shard(args, vocab_size):
    shard_id, shard = args
    enc = Tokenizer()
    all_tokens = []
    if shard.endswith('.json'):
        with open(shard, "r") as f:
            data = json.load(f)
            f.close()
        for example in tqdm(data['rows'], position=shard_id):
            text = example['row'][f"{data['features'][0]['name']}"]
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        with open(shard, "r") as f:
            for line in tqdm(f, position=shard_id):
                text = line.strip()
                tokens = enc.encode(text, bos=True, eos=False)
                all_tokens.extend(tokens)
            f.close()
        tokenized_filename = shard.replace(".txt", ".bin")
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # write to disk
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    print(f"Saved {tokenized_filename}")


def pretokenize(dataset_id, vocab_size):
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, f"{dataset_id}")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.*")))

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, dataset_id, split, max_seq_len):
        super().__init__()
        self.dataset_id = dataset_id
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        data_dir = os.path.join(DATA_CACHE_DIR, self.dataset_id)
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:

    @staticmethod
    def iter_batches(dataset_id, split, batch_size, max_seq_len, device, num_workers=0):
        ds = PretokDataset(dataset_id, split, max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, default="tiny_dataset", help="the id of underlying dataset.")
    parser.add_argument("stage", type=str, choices=["download", "train_tokenizer", "pretokenize"])
    parser.add_argument("--file_url", type=str, default="", help="download a file to disk from a given url.")
    parser.add_argument("--file_type", type=str, default="txt", help="download a file to disk in a given type.")
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download(args.dataset, args.file_url, args.file_type)
    elif args.stage == "pretokenize":
        pretokenize(args.dataset, args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")

