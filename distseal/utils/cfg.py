# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import torch

from distseal.augmentation.augmenter import get_dummy_augmenter
from distseal.models import VideoWam, build_embedder, build_extractor, build_baseline
from distseal.modules.jnd import JND, VarianceBasedJND


@dataclass
class SubModelConfig:
    """Configuration for a sub-model."""
    model: str
    params: DictConfig


@dataclass
class VideoWamConfig:
    """Configuration for a Video Seal model."""
    args: DictConfig
    embedder: SubModelConfig
    extractor: SubModelConfig


def resolve_config_path(cfg_path):
    # Resolve the config path in the following order:
    # 1. Search from the working directory
    # 2. Search from the source code directory
    if not Path(cfg_path).is_file():
        cfg_path = Path(__file__).parents[2].joinpath(cfg_path)

    return cfg_path


def message_to_tensor(message: str) -> str:
    """Convert a string message to a tensor.
    This is useful for converting messages back to tensors for processing
    """
    if not isinstance(message, str):
        raise TypeError(f"Expected a string, got {type(message)}")

    # Split the message by whitespace and convert to float
    return torch.tensor([float(x) for x in message.split()], dtype=torch.float32)


def get_config_from_checkpoint(ckpt_path: Path) -> VideoWamConfig:
    """
    Load configuration from a checkpoint file.

    Args:
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    VideoWamConfig: Loaded configuration.
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    args = checkpoint['args']
    args = OmegaConf.create(args)

    if not isinstance(args, DictConfig):
        raise Exception("Expected logfile to contain params dictionary.")

    # Load sub-model configurations
    embedder_cfg_path = resolve_config_path(args.embedder_config)
    embedder_cfg = OmegaConf.load(embedder_cfg_path)
    extractor_cfg_path = resolve_config_path(args.extractor_config)
    extractor_cfg = OmegaConf.load(extractor_cfg_path)

    # Create sub-model configurations
    embedder_model = args.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[embedder_model]
    extractor_model = args.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[extractor_model]

    return VideoWamConfig(
        args=args,
        embedder=SubModelConfig(model=embedder_model, params=embedder_params),
        extractor=SubModelConfig(model=extractor_model, params=extractor_params),
    )


def setup_model(config: VideoWamConfig, ckpt_path: Path):
    """
    Set up a VideoWam model from a configuration and checkpoint file.

    Args:
    config (VideoWamConfig): Model configuration.
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    VideoWam: Loaded model.
    """
    args = config.args

    # prepare some args for backward compatibility
    if "img_size_proc" in args:
        args.img_size = args.img_size_proc
    else:
        args.img_size = args.img_size_extractor

    if "hidden_size_multiplier" in args:
        args.hidden_size_multiplier = args.hidden_size_multiplier
    else:
        args.hidden_size_multiplier = 2
    
    # Build joint embedder-extractor (e.g. for training)
    embedder = build_embedder(config.embedder.model, config.embedder.params, args.nbits, args.hidden_size_multiplier)
    extractor = build_extractor(config.extractor.model, config.extractor.params, args.img_size, args.nbits)
    augmenter = get_dummy_augmenter()  # does nothing

    # Build attenuation
    if args.attenuation.lower().startswith("jnd"):
        attenuation_cfg = OmegaConf.load(args.attenuation_config)
        attenuation = JND(**attenuation_cfg[args.attenuation])
    elif args.attenuation.lower().startswith("simplified"):
        attenuation_cfg = OmegaConf.load(args.attenuation_config)
        attenuation = VarianceBasedJND(**attenuation_cfg[args.attenuation])
    else:
        attenuation = None

    # Build the complete model
    wam = VideoWam(
        embedder,
        extractor,
        augmenter,
        attenuation=attenuation,
        scaling_w=args.scaling_w,
        scaling_i=args.scaling_i,
        img_size=args.img_size,
        chunk_size=args.videowam_chunk_size,
        step_size=args.videowam_step_size
    )

    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        msg = wam.load_state_dict(checkpoint['model'], strict=False)
        # print(f"Model loaded successfully from {ckpt_path} with message: {msg}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    return wam

def setup_model_from_checkpoint(ckpt_path: str) -> VideoWam:
    """
    # Example usage
    ckpt_path = '/path/to/distseal/checkpoint.pth'
    wam = setup_model_from_checkpoint(ckpt_path)

    or 
    ckpt_path = 'baseline/wam'
    wam = setup_model_from_checkpoint(ckpt_path)
    """
    # load baselines. Should be in the format of "baseline/{method}"
    if "baseline" in ckpt_path:
        method = ckpt_path.split('/')[-1]
        return build_baseline(method)
    # load distseal model card
    elif ckpt_path.startswith('distseal'):
        return setup_model_from_model_card(ckpt_path)
    # load distseal checkpoints
    else:
        config = get_config_from_checkpoint(ckpt_path)
        return setup_model(config, ckpt_path)


