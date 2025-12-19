# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Sampling scripts for DC-AE on ImageNet.

Copyright (2025) mit-han-lab

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.


torchrun --nnodes=1 --nproc_per_node=8 scripts/sample_imagenet_dcae.py \
    diffusion_model=cards/uvit-h_distseal.yaml \
    autoencoder_model=cards/dc-ae-f64c128_base.yaml \
    samples_dir="distilled_dcae/gen_50k" \
    samples_per_class=50 \
    batch_size=16 \
    guidance_scale=1
"""

import os
from PIL import Image

import atexit
import torch
import numpy as np

from dataclasses import dataclass
from omegaconf import OmegaConf

from distseal.loader import load_generator, load_decoder
from distseal.modules.lora_helper import apply_lora_to_linear

from deps.efficientvit.apps.utils.dist import (
    dist_init, dist_destroy, dist_barrier, get_dist_rank, get_dist_size, is_master
)

@dataclass
class GenerateWMImagesConfig:
    diffusion_model: str  # diffusion model card
    autoencoder_model: str  # autoencoder model card
    samples_dir: str = "samples"  # output directory for generated samples
    samples_per_class: int = 50
    batch_size: int = 16
    classes: str = "imagenet"  # comma-separated classes to sample from, if imagenet uses all 1000 ImageNet classes
    guidance_scale: float = 1.0  # guidance scale, 1.0 means no guidance
    scaling_factor: float = 0.2889  # scaling factor for dc_ae latent space

    # LoRA options
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    
    # EMA options
    use_ema: bool = True  # use EMA weights if available in checkpoint
    ema_decay: float = 0.9998  # decay rate to match when loading EMA weights

def cleanup_process_group():
    """Cleanup function to properly destroy the process group."""
    dist_destroy()

def main(seed=1):
    cfg: GenerateWMImagesConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(GenerateWMImagesConfig), OmegaConf.from_cli())
    )

    # Suppress common warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant parameter should be passed explicitly.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*None of the inputs have requires_grad=True.*")

    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    # Distributed initialization
    dist_init()
    
    # Register cleanup function to be called at exit
    atexit.register(cleanup_process_group)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank = get_dist_rank()
    world_size = get_dist_size()
    torch.cuda.set_device(rank)

    # Load model from config
    diffusion_model = load_generator(cfg.diffusion_model, device=device)
    autoencoder_model = load_decoder(cfg.autoencoder_model, device=device)

    if cfg.use_lora:
        apply_lora_to_linear(diffusion_model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_linear_modules=cfg.lora_target_modules)


    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(seed + rank)

    if cfg.classes == "imagenet":
        all_classes = [i for i in range(1000)]
    else:
        cls_str = cfg.classes
        assert not cls_str.endswith(","), 'class string should not end with a ","'
        all_classes = [int(c) for c in cls_str.split(",")]

    samples_per_class = cfg.samples_per_class
    batch_size = cfg.batch_size

    if is_master():
        os.makedirs(cfg.samples_dir, exist_ok=True)
    dist_barrier()

    # Split class indices among ranks
    classes_per_rank = np.array_split(all_classes, world_size)[rank]

    for class_idx in classes_per_rank:
        class_dir = os.path.join(cfg.samples_dir, str(class_idx))
        os.makedirs(class_dir, exist_ok=True)
        num_generated = 0
        while num_generated < samples_per_class:
            current_batch = min(batch_size, samples_per_class - num_generated)
            prompts = torch.full((current_batch,), class_idx, dtype=torch.int, device=device)
            prompts_null = 1000 * torch.ones((current_batch,), dtype=torch.int, device=device)
            latent_samples = diffusion_model.generate(prompts, prompts_null, cfg.guidance_scale, eval_generator)
            latent_samples = latent_samples / cfg.scaling_factor

            # decode
            image_samples = autoencoder_model.decode(latent_samples)

            image_samples_uint8 = torch.clamp(127.5 * image_samples + 128.0, 0, 255).to(dtype=torch.uint8)
            image_samples_uint8 = image_samples_uint8.permute(0, 2, 3, 1).cpu().numpy()

            for i in range(current_batch):
                img_path = os.path.join(class_dir, f"{num_generated + i:04d}.png")
                Image.fromarray(image_samples_uint8[i], "RGB").save(img_path)
    
            num_generated += current_batch
            print(f"Rank {rank} - Class {class_idx}: {num_generated}/{samples_per_class} samples generated.")

    dist_barrier()
    if is_master():
        print("All samples generated across all GPUs.")

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print("Program interrupted. Cleaning up...")
    except Exception as e:
        print(f"Program encountered an error: {e}")
        raise
    finally:
        # Ensure cleanup happens even if atexit doesn't trigger
        cleanup_process_group()
