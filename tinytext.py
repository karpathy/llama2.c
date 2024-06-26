"""
Preprocess and serve a TinyText dataset as a DataLoader.

Follows the same interface as the TinyStories dataset.
"""

import argparse
import os
import random

import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
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


def download():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    data_filename = os.path.join(DATA_CACHE_DIR, "tinytext.txt")
    if not os.path.exists(data_filename):
        print(f"Missing {data_filename}...")
    else:
        print(f"{data_filename} exists")

    print("Download done.")

def pretokenize():
    enc = Tokenizer()

    data_file = os.path.join(DATA_CACHE_DIR, "tinytext.txt")

    all_tokens = []
    with open(data_file, "r") as f:
        lines = ['']
        for line in f:
            text = line.strip()
            if len(text) == 0:
                tokens = enc.encode('\n'.join(lines), bos=True, eos=False)
                lines = ['']
                all_tokens.extend(tokens)
            else:
                lines.append(text)
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens)}")
    with open(data_file.replace(".txt", ".bin"), "wb") as f:
        f.write(all_tokens.tobytes())
    print(f"Saved {data_file.replace('.txt', '.bin')}")
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

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
        data_file = os.path.join(DATA_CACHE_DIR, "tinytext.bin")
        m_all = np.memmap(data_file, dtype=np.uint16, mode="r")

        # split out 10% of the data for validation
        split_ix = int(len(m_all) * 0.9)
        if self.split == "train":
            m = m_all[:split_ix]
        else:
            m = m_all[split_ix:]

        num_batches = len(m) // self.max_seq_len
        num_batches -= 1  # drop the last partial batch
        assert num_batches > 0, "this split is way too small? investigate."

        while True:
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


class TinyTextTask:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_tokenizer", "pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    fun = {
        "download": download,
        "pretokenize": pretokenize,
    }
    fun[args.stage]()
