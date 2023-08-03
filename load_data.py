# Generic dataset loader: use it to download and pretokenize any dataset from Hugging Face
# Made by Leszek Mielnikow

# To download the dataset use the name:
#    $: python load_data.py download --dataset_name roneneldan/TinyStories

# To pretokenize the dataset using the GPT-2 tokenizer:
#    $: python load_data.py pretokenize --dataset_name roneneldan/TinyStories --tokenizer_name gpt2 --max_seq_len 512


import argparse
import os
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

DATA_CACHE_DIR = "data"

def download(dataset_name: str):
    """Downloads the dataset to disk."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Save the dataset to a file
    dataset_filename = os.path.join(DATA_CACHE_DIR, f"{dataset_name}.json")
    dataset.save_to_disk(dataset_filename)
    print(f"Dataset {dataset_name} downloaded and saved to {dataset_filename}.")


def pretokenize(dataset_name: str, tokenizer_name: str, max_seq_len: int):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def process_document(example):
        text = example["text"]
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_seq_len)
        # convert to uint16 nparray
        all_tokens = np.array(tokens, dtype=np.uint16)
        # write to disk
        tokenized_filename = os.path.join(DATA_CACHE_DIR, f"{dataset_name}_tokenized.bin")
        with open(tokenized_filename, "ab") as f:
            f.write(all_tokens.tobytes())

    # Tokenize and save the dataset to disk
    tokenized_filename = os.path.join(DATA_CACHE_DIR, f"{dataset_name}_tokenized.bin")
    if os.path.exists(tokenized_filename):
        print(f"{tokenized_filename} already exists, skipping tokenization...")
    else:
        # Iterate over the dataset and tokenize each example
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_document, example) for example in dataset["train"]]
            pbar = tqdm(total=len(futures), desc="Tokenizing")
            for future in as_completed(futures):
                pbar.update(1)
            pbar.close()

    print("Tokenization done.")

class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, dataset_name: str, split: str, max_seq_len: int):
        super().__init__()
        self.dataset_name = dataset_name
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
        tokenized_filename = os.path.join(DATA_CACHE_DIR, f"{self.dataset_name}_tokenized.bin")
        m = np.memmap(tokenized_filename, dtype=np.uint16, mode="r")
        num_batches = len(m) // self.max_seq_len
        num_batches -= 1  # drop the last partial batch
        assert num_batches > 0, "The dataset is too small? Please investigate."
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
    def iter_batches(dataset_name: str, split: str, batch_size: int, max_seq_len: int, device, num_workers: int = 0):
        ds = PretokDataset(dataset_name, split, max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize"])
    parser.add_argument("--dataset_name", type=str, default="tinystories")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download(args.dataset_name)
    elif args.stage == "pretokenize":
        pretokenize(args.dataset_name, args.tokenizer_name, args.max_seq_len)
