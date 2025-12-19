import functools
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Union
import torch
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader, is_image_file
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms

import linecache
import json
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


@functools.lru_cache()
def get_image_paths(root_path: str, pattern: Optional[str] = None, sorted_by_index: bool = False, num_samples: Optional[int] = None) -> List[str]:
    if pattern is None:
        paths = []
        for path, _, files in os.walk(root_path):
            for filename in files:
                paths.append(os.path.join(path, filename))
    else:
        paths = list(map(str, Path(root_path).glob(pattern)))
    image_fns = [fn for fn in paths if is_image_file(fn)]
    if sorted_by_index:
        sort_func = lambda x: int(Path(x).stem.split('_')[-1].split('.')[0])
    else:
        sort_func = lambda x: x
    res = sorted(image_fns, key=sort_func)
    if num_samples:
        res = res[:num_samples]
    return res


class ImageTransform:
    def __init__(self,
                 resize_shorter_edge: int = 256,
                 crop_size: int = 256,
                 random_crop: bool = True,
                 random_flip: bool = True,
                 normalize_mean: List[float] = [0., 0., 0.],
                 normalize_std: List[float] = [1., 1., 1.]):
        """Initializes the WebDatasetReader with specified augmentation parameters.

        Args:
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
        
        Raises:
            NotImplementedError: If the interpolation mode is not one of ["bicubic", "bilinear"].
        """
        train_transform = []
        interpolation = transforms.InterpolationMode.BICUBIC

        train_transform.append(
            transforms.Resize(resize_shorter_edge, interpolation=interpolation, antialias=True))
        if random_crop:
            train_transform.append(transforms.RandomCrop(crop_size))
        else:
            train_transform.append(transforms.CenterCrop(crop_size))
        if random_flip:
            train_transform.append(transforms.RandomHorizontalFlip())
        train_transform.append(transforms.ToTensor())
        # normalize_mean = [0, 0, 0] and normalize_std = [1, 1, 1] will normalize images into [0, 1],
        # normalize_mean = [0.5, 0.5, 0.5] and normalize_std = [0.5, 0.5, 0.5] will normalize images into [-1, 1].
        train_transform.append(transforms.Normalize(normalize_mean, normalize_std))

        self.train_transform = transforms.Compose(train_transform)
        self.eval_transform = transforms.Compose(
            [
                # Note that we always resize to crop_size during eval to ensure the results
                # can be compared against reference numbers on ImageNet etc.
                transforms.Resize(crop_size, interpolation=interpolation, antialias=True),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ]
        )
        print(f"self.train_transform: {self.train_transform}")
        print(f"self.eval_transform: {self.eval_transform}")


class ImageFolderWithoutClasses(BaseDataset):
    def __init__(
        self, 
        path: str,
        loader: Callable = default_loader,
        sorted_by_index: bool = True,
        file_pattern: Optional[str] = None,
        num_samples: Optional[int] = None
    ):
        # Main / original images
        self.samples = get_image_paths(path, pattern=file_pattern, sorted_by_index=sorted_by_index, num_samples=num_samples)

        self.loader = loader
        self.transform = transforms.ToTensor()
        
        if num_samples is not None:
            assert num_samples <= len(self.samples), f"There are {len(self.samples)} images in {path}, but configured num_samples = {num_samples}"
            self.num_samples = num_samples
        else:
            self.num_samples = len(self.samples)
        
        if self.num_samples == 0:
            if file_pattern is None:
                file_pattern = "*"
            raise ValueError(f"No images found in {path} with pattern {file_pattern}")

    def __getitem__(self, idx):
        img = self.loader(self.samples[idx])
        img = self.transform(img)
        return idx, img

    def __len__(self):
        return self.num_samples

def collate_even(batch) -> torch.Tensor:
    """
    Collate function that does not perform any padding, since the data is already evenly sized.
    Returns a tuple of indices and the data as a tensor.
    """
    indices = [idx for idx, _ in batch]
    data = [item for _, item in batch]
    return torch.tensor(indices, dtype=torch.int32), torch.stack(data)


class ImageFolderWithFilename(datasets.ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        num_samples: Optional[int] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )
        if num_samples is not None:
            self.num_samples = num_samples

    def __len__(self) -> int:
        if hasattr(self, 'num_samples'):
            return self.num_samples
        return super().__len__()

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class PretokenizedDataSetJSONL(BaseDataset):
    def __init__(self, data_path):
        super().__init__()
        self.jsonl_file = data_path
        self.num_lines = sum(1 for _ in open(self.jsonl_file))
        # Ensure the file is cached
        linecache.checkcache(self.jsonl_file)
        print("Number of data:", self.num_lines)

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        line = linecache.getline(self.jsonl_file, idx + 1).strip()
        data = json.loads(line)
        return torch.tensor(data["class_id"]), torch.tensor(data["tokens"])
