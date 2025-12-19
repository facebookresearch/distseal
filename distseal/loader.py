# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Toolkit for loading different DistSeal model files

The code is adopted from multiple codebases, with copyright notes below.


========================================================================
RAR:  Randomized Autoregressive Visual Generation

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 


========================================================================
EfficientVIT: Efficient Vision Foundation Models for High-Resolution Generation
and Perception

Copyright [2023] [Han Cai]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

from distseal.modules.lora_helper import apply_lora
from distseal.utils.checkpoint_manager import resolve_checkpoint
from distseal.utils.cfg import message_to_tensor, get_config_from_checkpoint, setup_model

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise NotImplementedError(
        "huggingface_hub is not found. Please install it to run the demo: `pip install huggingface_hub`"
    )



rar_gen_params = {
    "rar_b": {
        "hidden_size": 768,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 3072,
        "randomize_temperature": 1.0,
        "guidance_scale": 16.0,
        "guidance_scale_pow": 2.75,
    },
    "rar_l": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "randomize_temperature": 1.02,
        "guidance_scale": 15.5,
        "guidance_scale_pow": 2.5,
    },
    "rar_xl": {
        "hidden_size": 1280,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "intermediate_size": 5120,
        "randomize_temperature": 1.02,
        "guidance_scale": 6.9,
        "guidance_scale_pow": 1.5,
    },
    "rar_xxl": {
        "hidden_size": 1408,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "intermediate_size": 6144,
        "randomize_temperature": 1.02,
        "guidance_scale": 8.0,
        "guidance_scale_pow": 1.2,
    },
}


distseal_detectors = {
    "maskgit-vq": "distseal_maskgit-vq-f16-256_after-quant.bin",
}

def get_rar_generator(cfg, device="cpu"):
    from deps.rar.modeling.rar import RAR
        
    # Build RAR object    
    config_file = Path(__file__).parent.parent.joinpath("configs/generation/rar.yaml")
    config = OmegaConf.load(config_file)
    
    hf_namespace = cfg.get("hf_namespace", None)
    ckpt = resolve_checkpoint(cfg.checkpoint, hf_namespace=hf_namespace)
        
    for param, value in rar_gen_params[cfg.size].items():
        setattr(config.model.generator, param, value)
    
    generator = RAR(config)
    
    if config.training.get("use_lora", False):
        apply_lora(generator, config.use_lora, check=True)
    
    # Load weight
    generator.load_state_dict(torch.load(ckpt, map_location="cpu"))
    generator.eval().to(device)
    generator.requires_grad_(False)
    generator.set_random_ratio(0)
    return generator


def get_diffusion_generator(cfg, device="cpu"):

    if cfg.get("checkpoint", None):
        hf_namespace = cfg.get("hf_namespace", None)
        ckpt = resolve_checkpoint(cfg.checkpoint, hf_namespace=hf_namespace)
    else:
        ckpt = None

    if cfg.model_arch == "dit":
        from deps.efficientvit.diffusioncore.models.dit import DiT as diffusion_cls

    elif cfg.model_arch == "uvit":
        from deps.efficientvit.diffusioncore.models.uvit import UViT as diffusion_cls
    
    else:
        raise NotImplementedError(f"Unsupported diffusion model architecture: {cfg.model_arch}")
        
    cfg.model.input_size = cfg.resolution // cfg.spatial_compression_ratio
    cfg.model.pretrained_path = ckpt
    model = diffusion_cls(cfg.model)
    
    model.eval().to(device)
    model.requires_grad_(False)
    return model


def parse_model_card(model_card: str):
    if not Path(model_card).exists():
        model_card = Path(__file__).parent.parent.joinpath("cards").joinpath(model_card)
    if not Path(model_card).exists():
        raise FileNotFoundError(
            f"Model card not found: {model_card}.\n"
            "Make sure your path is correct, or place it in the 'card' directory."
        )
    conf = OmegaConf.load(model_card)
    return conf


def load_generator(model_card: str, device: str = "cpu"):
    conf = parse_model_card(model_card)

    if conf.model_arch == "rar":
        return get_rar_generator(conf, device=device)
    else:
        return get_diffusion_generator(conf, device=device)


def get_maskgit_tokenizer(cfg, device: str = "cpu"):  
    from deps.rar.modeling.titok import PretrainedTokenizer
    
    hf_namespace = cfg.get("hf_namespace", None)
    ckpt = resolve_checkpoint(cfg.checkpoint, hf_namespace=hf_namespace)

    tokenizer = PretrainedTokenizer(ckpt)
    tokenizer.eval().to(device)
    return tokenizer


def get_dcae_autoencoder(cfg, device="cpu"):
    from deps.efficientvit.ae_model_zoo import create_dc_ae_model_cfg, DCAE_HF
    from deps.efficientvit.models.efficientvit.dc_ae import DCAE
    
    if cfg.get("checkpoint", None):
        hf_namespace = cfg.get("hf_namespace", None)
        ckpt = resolve_checkpoint(cfg.checkpoint, hf_namespace=hf_namespace)
        cfg = create_dc_ae_model_cfg(cfg.base, ckpt)
        autoencoder = DCAE(cfg).eval().to(device)
    else:
        autoencoder = DCAE_HF.from_pretrained(f"mit-han-lab/{cfg.name}").eval().to(device) 

    return autoencoder


def load_decoder(model_card: str, device: str = "cpu"):
    conf = parse_model_card(model_card)
    if conf.model_arch == "maskgit":
        return get_maskgit_tokenizer(conf, device=device)
    
    elif conf.model_arch == "dc-ae":
        return get_dcae_autoencoder(conf, device=device)
    
    else:
        raise NotImplementedError(f"Unsupported decoder model architecture: {conf.model_arch}")


def load_message(msg_path: str) -> torch.Tensor:
    if msg_path.endswith(".txt"):
        with open(msg_path, "r") as f:
            message = f.read().strip()
            msg_tensor = message_to_tensor(message).unsqueeze(0)
        return msg_tensor
    msg = np.load(msg_path)
    msg = torch.from_numpy(msg).long()
    if msg.ndim < 2:
        msg = msg.unsqueeze(0)
    return msg


def load_detector(model_card: Path | str, device: str = "cpu"):
    """
    Set up the DistSeal extractor model from a model card YAML file.
    Args:
        model_card (Path | str): Path to the model card YAML file or name of the model card.
    Returns:
        Extractor: Loaded extractor model.
    """
    card_conf = parse_model_card(model_card)

    ckpt_path = resolve_checkpoint(card_conf.checkpoint)
    config = get_config_from_checkpoint(ckpt_path)
    wm_model = setup_model(config, ckpt_path)
    wm_model.eval().to(device)
    
    message = None
    if card_conf.get("message", None):
        message_file = resolve_checkpoint(card_conf.message)
        message = load_message(message_file).to(device)

    return wm_model.detector, message


def load_watermarker(model_card: Path | str, device: str = "cpu"):
    """
    Set up the post-hoc watermarker from a model card YAML file.
    Args:
        model_card (Path | str): Path to the model card YAML file or name of the model card.
    Returns:
        VideoWam: Loaded model.
    """
    card_conf = parse_model_card(model_card)

    ckpt_path = resolve_checkpoint(card_conf.checkpoint)
    config = get_config_from_checkpoint(ckpt_path)
    wm_model = setup_model(config, ckpt_path)
    wm_model.eval().to(device)
    
    message = None
    if card_conf.get("message", None):
        message_file = resolve_checkpoint(card_conf.message)
        message = load_message(message_file).to(device)

    return wm_model, message, card_conf.get("scaling_w", None)
