# Copyright (c) Meta Platforms, Inc. and affiliates.
from .sequential import Sequential
from .geometric import Crop, HorizontalFlip, Identity, Perspective, Resize, Rotate
from .valuemetric import JPEG, Brightness, Contrast, GaussianBlur, Grayscale, Hue, MedianFilter, Saturation
from .neuralcompression import VQGAN1024, VQGAN16384, StableDiffusionVAE, BMSHJ2018Factorized, BMSHJ2018Hyperprior, MBT2018Mean, MBT2018, Cheng2020Anchor, Cheng2020Attn

def get_validation_augs_subset() -> list:
    """
    Get the validation augmentations.
    """
    augs = [
        (Identity(),                [0]),  # No parameters needed for identity
        (HorizontalFlip(),          [0]),  # No parameters needed for flip
        (Crop(),                    [0.71]),  # size ratio
        (Brightness(),              [0.5]),
        (JPEG(),                    [60]),
        # (VQGAN16384(),              [0]),  # No parameters needed for VQGAN
        (Sequential(JPEG(), Crop(), Brightness()), [(60, 0.71, 0.5)]),
    ]
    return augs


def get_validation_augs(
    only_identity: bool = False,
    only_combined: bool = False
) -> list:
    """
    Get the validation augmentations.
    Args:
        only_identity (bool): Whether to only use identity augmentation
        only_combined (bool): Whether to only use combined augmentations
    """
    if only_identity:
        augs = [
            (Identity(),          [0]),  # No parameters needed for identity
        ]
    elif only_combined:
        augs = [
            (Identity(),          [0]),  # Always include identity for baseline
            (Sequential(JPEG(), Crop(), Brightness()), [(40, 0.71, 0.5)]),
        ]
    else:
        augs = [
            (Identity(),          [0]),  
            (HorizontalFlip(),    [0]),  
            (Rotate(),            [5, 10, 30, 45, 90]),  
            (Resize(),            [0.32, 0.45, 0.55, 0.63, 0.71, 0.77, 0.84, 0.89, 0.95, 1.00]),  # size ratio, such 0.1 increment in area ratio
            (Crop(),              [0.32, 0.45, 0.55, 0.63, 0.71, 0.77, 0.84, 0.89, 0.95, 1.00]),  
            (Perspective(),       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),  # distortion_scale
            (Brightness(),        [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            (Contrast(),          [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            (Saturation(),        [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            (Hue(),               [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            (Grayscale(),         [-1]),  # No parameters needed for grayscale
            (JPEG(),              [40, 50, 60, 70, 80, 90]),
            (GaussianBlur(),      [3, 5, 9, 13, 17]),
            # (VQGAN1024(),         [0]),  # No parameters needed for VQGAN
            # (VQGAN16384(),        [0]),  # No parameters needed for VQGAN
            (Sequential(JPEG(), Crop(), Brightness()), [(40, 0.71, 0.5)]),
            (Sequential(JPEG(), Crop(), Brightness()), [(60, 0.71, 0.5)]),
            (Sequential(JPEG(), Crop(), Brightness()), [(80, 0.71, 0.5)]),
        ]
    return augs
