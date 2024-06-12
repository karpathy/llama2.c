import abc

import glob
import json
import os

from typing import List, Iterator

import requests
from tqdm import tqdm


class Dataset(abc.ABC):

    @abc.abstractmethod
    def download(self):
        """Download, unpack and custom preprocess dataset to local directory"""
        pass

    @abc.abstractmethod
    def list_files(self) -> List[str]:
        """
        Returns a list of files belonging to the dataset.
        These are raw files that belong to the original dataset distributive. Not are pre-tokenized files.
        """
        raise NotImplemented()

    @abc.abstractmethod
    def examples_of(self, filepath: str) -> Iterator[str]:
        """
        Returns an iterator over examples of specified file.
        Should return one example per iteration.
        List of files can be got by self.files() method.
        """
        raise NotImplemented()


class TinyStories(Dataset):
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"

    def __init__(self, cache_dir: str = "data") -> None:
        super().__init__()
        self.cache_dir = cache_dir

        # Directory where unpacked data will be stored
        self.data_dir = os.path.join(self.cache_dir, "TinyStories_all_data")

    def download(self):
        """Downloads the TinyStories dataset to self.cache_dir"""
        os.makedirs(self.cache_dir, exist_ok=True)

        # download the TinyStories dataset, unless it's already downloaded
        data_filename = os.path.join(self.cache_dir, "TinyStories_all_data.tar.gz")
        if not os.path.exists(data_filename):
            print(f"Downloading {self.data_url} to {data_filename}...")
            self.download_file(self.data_url, data_filename)
        else:
            print(f"{data_filename} already exists, skipping download...")

        # unpack the tar.gz file into all the data shards (json files)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Unpacking {data_filename}...")
            os.system(f"tar -xzf {data_filename} -C {self.data_dir}")
        else:
            print(f"{self.data_dir} already exists, skipping unpacking...")

        # print a single example just for debugging and such
        shard_filenames = self.list_files()
        with open(shard_filenames[0], "r") as f:
            data = json.load(f)
        print("Download done.")
        print(f"Number of shards: {len(shard_filenames)}")
        print(f"Example story:\n{data[0]}")

    def download_file(self, url: str, fname: str, chunk_size=1024):
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

    def list_files(self) -> List[str]:
        """
        Return list of files.
        That is a raw files that belong to original dataset distributive. Not a pretokenized files.
        """
        return sorted(glob.glob(os.path.join(self.data_dir, "*.json")))

    def examples_of(self, filepath: str) -> Iterator[str]:
        """
        Should return one example per iteration for specified files.
        List of files can be got by self.files() method.
        """
        with open(filepath, mode='r') as infile:
            data = json.load(infile)

        for row in data:
            yield row['story']


class SQLCreateContext(Dataset):
    data_url = "https://huggingface.co/datasets/b-mc2/sql-create-context/resolve/main/sql_create_context_v4.json"

    def __init__(self, cache_dir: str = "data") -> None:
        super().__init__()
        self.cache_dir = cache_dir

        # Directory where unpacked data will be stored
        self.data_dir = os.path.join(self.cache_dir, "sql_create_context")

    def download(self):
        """Downloads the dataset to self.data_dir"""
        os.makedirs(self.cache_dir, exist_ok=True)

        # download dataset, unless it's already downloaded
        data_filename = os.path.join(self.cache_dir, os.path.basename(self.data_url))
        if not os.path.exists(data_filename):
            print(f"Downloading {self.data_url} to {data_filename}...")
            self.download_file(self.data_url, data_filename)
        else:
            print(f"{data_filename} already exists, skipping download...")

        # Original dataset has only 1 file. Split it on 10 shards.
        with open(data_filename) as infile:
            data = json.load(infile)
        os.makedirs(self.data_dir, exist_ok=True)
        files = [open(os.path.join(self.data_dir, f'data_{x:02d}.jsonl'), mode='w') for x in range(10)]
        for idx, example in enumerate(data):
            files[idx % 10].write(json.dumps(example) + '\n')
        [f.close() for f in files]

        # print a single example just for debugging and such
        shard_filenames = self.list_files()
        with open(shard_filenames[0], "rt") as f:
            data = [json.loads(line) for line in f]
        print("Download done.")
        print(f"Number of shards: {len(shard_filenames)}")
        print(f"Example story:\n{data[0]}")

    def download_file(self, url: str, fname: str, chunk_size=1024):
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

    def list_files(self) -> List[str]:
        """
        Return list of files.
        That is a raw files that belong to original dataset distributive. Not a pretokenized files.
        """
        return sorted(glob.glob(os.path.join(self.data_dir, "*.jsonl")))

    def examples_of(self, filepath: str) -> Iterator[str]:
        """
        Should return one example per iteration for specified files.
        List of files can be got by self.files() method.
        """
        with open(filepath, mode='r') as infile:
            data = [json.loads(line) for line in infile]


        for row in data:
            yield f"{row['question']}. Use the following DDL: {row['context']}. Write an SQL query. Here is possible SQL query: {row['answer']}"