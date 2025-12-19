# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download

import torch.distributed as dist
import distseal.utils.dist as udist


def verify_file_integrity(local_file, remote_url, chunk_size=8192):
    """
    Verify that a local file matches the remote file by comparing hashes.
    
    Args:
        local_file: Path to the local file
        remote_url: URL of the remote file
        hf_namespace: HuggingFace namespace if it's a HF URL
        chunk_size: Chunk size for reading the local file
    
    Returns:
        True if hashes match, False otherwise
    """
    if not os.path.exists(local_file):
        print(f"Local file {local_file} does not exist")
        return False
    
    # Compute local file hash
    local_hash = hashlib.md5()
    with open(local_file, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            local_hash.update(chunk)
    local_md5 = local_hash.hexdigest()
    
    # Try to get hash from HTTP headers
    response = requests.head(remote_url, allow_redirects=True, timeout=10)
    response.raise_for_status()
    
    # Try different header fields
    remote_hash = (
        response.headers.get('content-md5') or 
        response.headers.get('etag', '').strip('"')
    )
    return local_md5 in remote_hash or remote_hash in local_md5


def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def maybe_download_checkpoint(url):

    basename = urlparse(url).path.split("/")
    filename = os.path.abspath(os.path.join("checkpoints", *basename))

    if os.path.exists(filename):
        if verify_file_integrity(filename, url):
            print(f"File {filename} exists and has matching hash, skipping download")
            return filename

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"File {url} downloaded successfully to {filename}")
    return filename


def resolve_checkpoint(checkpoint_path, hf_namespace=None) -> str:
    if Path(checkpoint_path).is_file():
        return checkpoint_path
            
    basename = urlparse(checkpoint_path).path.split("/")
    filename = os.path.abspath(os.path.join("checkpoints", *basename))

    if os.path.exists(filename):        
        print(f"File {filename} exists, skipping download")
        return filename

    if hf_namespace:        
        # Download the checkpoint to local checkpoints directory
        local_dir = os.path.dirname(filename)
        os.makedirs(local_dir, exist_ok=True)
        
        if udist.is_dist_avail_and_initialized():
            if udist.is_main_process():
                # download only on the main process
                hf_hub_download(
                    repo_id=hf_namespace,
                    filename=str(checkpoint_path),
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
            dist.barrier()
        else:
            hf_hub_download(
                repo_id=hf_namespace,
                filename=str(checkpoint_path),
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )

    elif is_url(checkpoint_path):
        if udist.is_dist_avail_and_initialized():
            if udist.is_main_process():
                maybe_download_checkpoint(checkpoint_path)
            dist.barrier()
        else:
            maybe_download_checkpoint(checkpoint_path)
    else:
        raise RuntimeError(f"Path or uri {checkpoint_path} is unknown or does not exist")
    
    return filename