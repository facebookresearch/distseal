# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import logging
import os


import torch
from PIL import Image

from torchvision.datasets.folder import is_image_file
from torchvision.transforms import ToTensor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_image_paths(path, cache_dir=None):
    if cache_dir is None:
        # FIXME: hardcoded cache dir
        cache_dir = os.getenv("DISTSEAL_CACHE_DIR", '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = path.replace('/', '_') + '.json'
    cache_file = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            paths = json.load(f)
    else:
        paths = []
        for root, _, files in os.walk(path):
            for filename in files:
                if is_image_file(filename):
                    paths.append(os.path.join(root, filename))
        paths = sorted(paths)
        with open(cache_file, 'w') as f:
            json.dump(paths, f)
    return paths


class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, mask_transform=None):
        # assuming 'path' is a folder of image files path and
        # 'annotation_path' is the base path for corresponding annotation json files
        self.samples = get_image_paths(path)
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = ToTensor()(img)

        if self.transform:
            img = self.transform(img)

        # Get MASKS
        mask = torch.ones_like(img[0:1, ...])

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        return img, mask

    def __len__(self):
        return len(self.samples)
