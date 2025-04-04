import tqdm
import argparse
import random
import json
import datasets
import requests
import math
import os
import gzip
import numpy as np
import multiprocessing
import huggingface_hub
import glob
import tempfile

from datasets import load_dataset, Dataset, DatasetDict, interleave_datasets
from streaming.base.util import _merge_index_from_root, merge_index
from transformers import set_seed, AutoTokenizer
from streaming import MDSWriter, StreamingDataset
from huggingface_hub import HfFileSystem

from data_utils import MDS_COLS_TEXT

set_seed(11111111)

FILES_INFO = None

ACCEPTED_SOURCE_FORMATS = ["hf_hub", "local_hf", "parquet"]
ACCEPTED_TARGET_FORMATS = ["hf", "parquet", "mds"]

def push_to_hub_incrementally(repo_name, local_path):
    api = huggingface_hub.HfApi()
    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo
    print(f"Uploading {local_path} to {repo_name}/{local_path}")
    try:
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            path_in_repo=local_path,
            repo_type="dataset",
            multi_commits=True,
            multi_commits_verbose=True,
        )
    except Exception as e:
        print(e)
        import time
        time.sleep(30)
        print(f"Error uploading {local_path} to {repo_name}, trying again")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            path_in_repo=local_path,
            repo_type="dataset",
            multi_commits=True,
            multi_commits_verbose=True,
        )

    os.system(f"rm -rf {local_path}")
    print(f"Pushed {local_path} to {repo_name}")


def load_hf_hub_dataset(repo_name, config_name=None, split_name=None, streaming=True) -> (
        datasets.Dataset | datasets.DatasetDict | datasets.IterableDataset | datasets.IterableDatasetDict):
    """
    Load a dataset from the Hugging Face Hub.

    Args:
        repo_name: Name of the dataset repository
        config_name: Optional configuration name
        split_name: Optional split name
        streaming: Whether to use streaming mode (default: True)

    Returns:
        The loaded dataset, which could be one of several types depending on parameters
    """
    # Load the dataset from the Hugging Face Hub
    if config_name is not None:
        dataset = datasets.load_dataset(repo_name, config_name, streaming=streaming)
    else:
        dataset = datasets.load_dataset(repo_name, streaming=streaming)

    # If a specific split is requested and available, return it
    if split_name and split_name in dataset:
        return dataset[split_name]

    # Otherwise return the entire dataset
    return dataset


def load_local_hf_dataset(local_dataset_path, split_name=None) -> datasets.Dataset | datasets.DatasetDict:
    """
    Load a local Hugging Face dataset.

    Args:
        local_dataset_path: Path to the local dataset
        split_name: Optional name of the split to return

    Returns:
        The loaded dataset or dataset dictionary
    """
    # Load the local hf dataset
    dataset = datasets.load_from_disk(local_dataset_path)

    # If a specific split is requested and available, return it
    if split_name and isinstance(dataset, datasets.DatasetDict) and split_name in dataset:
        return dataset[split_name]

    # Otherwise return the entire dataset (either a Dataset or DatasetDict with all splits)
    return dataset


def load_local_parquet_dataset(parquet_path, split_name=None) -> datasets.Dataset | datasets.DatasetDict:
    """
    Load a Hugging Face dataset from parquet file(s).

    Args:
        parquet_path: Path to a parquet file or directory containing parquet files
        split_name: Optional name of the split to return

    Returns:
        The loaded HF dataset or HF dataset dictionary
    """
    # Load the dataset from parquet file(s)
    # If parquet_path is a directory, this will load all parquet files as a single dataset
    dataset = datasets.load_dataset("parquet", data_files=parquet_path)

    # If a specific split is requested and available, return it
    if split_name and isinstance(dataset, datasets.DatasetDict) and split_name in dataset:
        return dataset[split_name]

    # Otherwise return the entire dataset (either a Dataset or DatasetDict with all splits)
    return dataset


def convert_to_hf(dataset, upload_repo=None, local_save_path=None):
    """
    Convert a dataset to Hugging Face format, optionally uploading it to the Hub and/or saving it locally.

    Args:
        dataset: A Dataset, DatasetDict, IterableDataset, or IterableDatasetDict
        upload_repo: Optional repository name to upload the dataset to the Hugging Face Hub
        local_save_path: Optional local path to save the dataset

    Returns:
        The dataset in non-streaming format (converted if necessary)
    """
    import os
    from huggingface_hub import HfApi
    import datasets

    # Convert iterable datasets to regular datasets if necessary
    if isinstance(dataset, (datasets.IterableDataset, datasets.IterableDatasetDict)):
        print("Converting streaming dataset to regular dataset...")
        if isinstance(dataset, datasets.IterableDataset):
            dataset = dataset.to_dataset()
        else:  # IterableDatasetDict
            dataset = datasets.DatasetDict({
                split: dataset[split].to_dataset() for split in dataset
            })


def sample_hf(source_format=None, target_format = None, upload_repo=None, repo_name=None, split_name=None, config_name=None, local_save_path=None,
              local_dataset_path=None):
    print(f"Sampling the data with repo {repo_name} and {split_name} and {config_name}...")

    if source_format == "hf_hub":
        dataset = load_hf_hub_dataset(repo_name, config_name=config_name, split_name=split_name)
    elif source_format == "local_hf":
        dataset = load_local_hf_dataset(local_dataset_path, split_name=split_name)
    elif source_format == "parquet":
        dataset = load_local_parquet_dataset(local_dataset_path, split_name=split_name)
    else:
        raise ValueError(f"Invalid source format: {source_format}. Must be one of: {ACCEPTED_SOURCE_FORMATS}.")

    if target_format == "hf":
        convert_to_hf(dataset, upload_repo=upload_repo, local_save_path=local_save_path)


    tmp_cache_dir = None
    config_name_dirsafe = config_name.replace("/", "-") if config_name is not None else "default"
    split_name_dirsafe = split_name.replace("/", "-") if split_name is not None else "default"
    tmp_cache_dir = f"{repo_name.replace('/', '-')}---{split_name_dirsafe}---{config_name_dirsafe}" if repo_name else "local_dataset"

    if not os.path.isfile(os.path.join(tmp_cache_dir, "index.json")):
        print(f"Writing to MDS...")
        with MDSWriter(out=tmp_cache_dir, columns=MDS_COLS_TEXT, compression='zstd') as train_writer:
            for idx, item in tqdm.tqdm(enumerate(dataset)):
                if 'id' not in item:
                    item['id'] = str(idx)  # Add a unique id if missing
                train_writer.write(item)

    if local_save_path:
        # Save locally
        local_save_dir = os.path.join(local_save_path, tmp_cache_dir)
        os.makedirs(local_save_dir, exist_ok=True)
        for file_name in os.listdir(tmp_cache_dir):
            full_file_name = os.path.join(tmp_cache_dir, file_name)
            if os.path.isfile(full_file_name):
                os.system(f"cp {full_file_name} {local_save_dir}")

        print(f"Saved locally to {local_save_dir}")

    if upload_repo:
        print(f"Pushing to HF...")
        dataset = StreamingDataset(local=tmp_cache_dir, shuffle=False, split=None, batch_size=1)
        num_instances = len(dataset)
        push_to_hub_incrementally(
            upload_repo,
            tmp_cache_dir
        )

        fs = HfFileSystem()
        size_of_folder = fs.du(f"datasets/{upload_repo}")
        with open("dataset_info.jsonl", "a") as f:
            f.write(json.dumps({"dataset": upload_repo, "split_name": split_name, "config_name": config_name,
                                "size": size_of_folder / 1e9, "instances": num_instances}) + "\n")


def validate_args(args):
    if args.source_format not in ["hf_hub", "local_hf", "parquet"]:
        raise ValueError(f"Invalid source format: {args.source_format}. Must be one of: {ACCEPTED_SOURCE_FORMATS}.")

    if args.target_format not in ["hf", "parquet", "mds"]:
        raise ValueError(f"Invalid target format: {args.target_format}. Must be one of: {ACCEPTED_TARGET_FORMATS}.")

    if args.source_format == args.target_format:
        raise ValueError(f"Source format and target format cannot be the same: {args.source_format}.")

    if args.source_format == "hf_hub" and not args.repo_name:
        raise ValueError("When source format is 'hf_hub', repo_name must be provided.")

    if args.source_format in ["local_hf", "local_hf", "parquet"] and not args.local_dataset_path:
        raise ValueError("When source format is a local dataset, local_dataset_path must be provided.")

    # Info about not uploading to HF hub (upload_repo not provided)
    if args.upload_repo is None:
        print("INFO: upload_repo is not provided. The dataset will not be uploaded to the Hugging Face Hub.")

    # Info about not saving locally (local_save_path not provided)
    if args.local_save_path is None:
        print("INFO: local_save_path is not provided. The dataset will not be saved locally.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Source (format) of the dataset to be converted
    parser.add_argument("-sf", "--source_format", type=str, required=True)
    # Destination format of the dataset to be converted
    parser.add_argument("-tf", "--target_format", type=str, required=True)
    # Possible formats: hf_hub, local_hf, parquet, mds

    # HF Repository name to upload to
    parser.add_argument("-u", "--upload_repo", type=str, required=False)
    # Hugging Face dataset repo name to be converted
    parser.add_argument("-r", "--repo_name", type=str, required=False)
    # Hugging Face dataset split to be converted
    parser.add_argument("-s", "--repo_split", type=str, required=False)
    # Hugging Face dataset config name to be converted
    parser.add_argument("-c", "--repo_config", type=str, required=False)
    # Local path to save the converted dataset
    parser.add_argument("-l", "--local_save_path", type=str, required=False)
    # Local path to the dataset to be converted
    parser.add_argument("-d", "--local_dataset_path", type=str, required=False)
    args = parser.parse_args()

    validate_args(args)

    args.source_format = ""
    args.target_format = ""
    args.upload_repo = None
    args.repo_name = "./c4_samples_local_repo"
    args.repo_split = "train"
    args.repo_config = None
    args.local_save_path = "./out_test"
    args.local_dataset_path = None

    sample_hf(args.source_format, args.target_format,
              args.upload_repo, args.repo_name, args.repo_split, args.repo_config,
              args.local_save_path, args.local_dataset_path)

    # example usage:
    #   python hf_to_mds.py -r HF_DATASET -c CONFIG -s SPLIT -u HF_SAVE_PATH -l LOCAL_SAVE_PATH -d LOCAL_DATASET_PATH
    #   python hf_to_mds_new.py -l ./out_test -d ./c4_samples_dataset_with_split -s train
    #   python hf_to_mds_new.py -l ./out_test -r ./c4_samples_local_repo -s train