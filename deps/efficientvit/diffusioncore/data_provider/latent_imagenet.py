import random
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import DatasetFolder

from deps.efficientvit.diffusioncore.data_provider.base import BaseDataProvider, BaseDataProviderConfig

__all__ = ["LatentImageNetDataProviderConfig", "LatentImageNetDataProvider"]


@dataclass
class LatentImageNetDataProviderConfig(BaseDataProviderConfig):
    name: str = "latent_imagenet"
    data_dir: str = "assets/data/latent/dc_ae_f32c32/imagenet_512"
    target_class_index: Optional[int] = None  # If not None, use only a single class from ImageNet for training
    class_num_samples: int = 50


class LatentImageNetDataProvider(BaseDataProvider):
    def __init__(self, cfg: LatentImageNetDataProviderConfig, seed: int):
        self.seed = seed # we need to set the seed here for having access to the seed in build_datasets

        super().__init__(cfg)
        self.cfg: LatentImageNetDataProviderConfig

    def build_datasets(self) -> tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        full_dataset = DatasetFolder(self.cfg.data_dir, np.load, [".npy"])
        if self.cfg.target_class_index is None:
            return full_dataset, None, None
        
        class_to_idx = full_dataset.class_to_idx
        assert (
            len(class_to_idx) == 1000
        ), f"Expected 1000 classes in ImageNet, but got {len(class_to_idx)} classes: {class_to_idx}"

        # Select a single class (e.g., class index 0)
        indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == self.cfg.target_class_index]
        g = torch.Generator()
        g.manual_seed(self.seed)
        permutation = torch.randperm(self.cfg.class_num_samples, generator=g).tolist()
        indices = [indices[i] for i in permutation]

        # Subset the dataset to only include samples from the target class
        subset_samples = [full_dataset.samples[i] for i in indices]
        subset_dataset = DatasetFolder(
            root=self.cfg.data_dir,
            loader=np.load,
            extensions=[".npy"],
        )
        subset_dataset.samples = subset_samples
        subset_dataset.targets = [s[1] for s in subset_samples]
        subset_dataset.class_to_idx = {list(class_to_idx.keys())[self.cfg.target_class_index]: self.cfg.target_class_index}
        subset_dataset.classes = [list(class_to_idx.keys())[self.cfg.target_class_index]]

        return subset_dataset, None, None