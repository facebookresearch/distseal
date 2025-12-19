# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torchvision
from torch import Tensor


def save_img(img: Tensor, out_path: str) -> None:
    """
    Saves an image tensor to a file.

    Args:
    img (Tensor): The image tensor with shape (C, H, W) where
                  C is the number of channels (should be 3),
                  H is the height,
                  W is the width.
    out_path (str): The output path for the saved image file.

    Raises:
    AssertionError: If the input tensor does not have the correct dimensions or channel size.
    """
    # Assert the image tensor has the correct dimensions
    assert img.dim() == 3, "Input image tensor must have 3 dimensions (C, H, W)"
    assert img.size(0) == 3, "Image tensor's channel size must be 3"

    # Clamp the values and convert to numpy
    img = img.clamp(0, 1) * 255
    img = img.to(torch.uint8).cpu()

    # Write the image file
    img_pil = torchvision.transforms.ToPILImage()(img)
    img_pil.save(out_path)
