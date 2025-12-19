"""Demo file for sampling images from TiTok.

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
"""

import contextlib
import os

import matplotlib.pyplot as plt

import numpy as np
import random

import torch
import torchvision.transforms as T


@contextlib.contextmanager
def kv_context(model):
    model.enable_kv_cache()
    try:
        yield
    finally:
        model.disable_kv_cache()


def set_seed(seed: int):
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              guidance_scale=3.0,
              guidance_decay="constant",
              guidance_scale_pow=3.0,
              randomize_temperature=2.0,
              softmax_temperature_annealing=False,
              num_sample_steps=8,
              device="cuda",
              return_tensor=False):
    generator.eval()
    tokenizer.eval()
    if labels is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        num_sample_steps=num_sample_steps,
    )
    
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image


def fill_image_with_mask(
    ids: torch.Tensor,
    image_seq_len: int,
    mask_value: float = 128,
):
    # Create a mask tensor with the same shape as the input ids
    mask = torch.full((ids.shape[0], image_seq_len), mask_value, device=ids.device)

    # Fill the mask with the input ids
    mask[:, :ids.shape[1]] = ids

    return mask


def gen_and_save_images(tokens, step, out_dir, tokenizer, prefix="step"):
    os.makedirs(out_dir, exist_ok=True)
    
    image_tensors = tokenizer.decode_tokens(
        tokens.view(tokens.shape[0], -1)
    )
    image_tensors = torch.clamp(image_tensors, 0.0, 1.0) * 255.0
    for i, img_tensor in enumerate(image_tensors):
        pil_img = T.functional.to_pil_image(img_tensor.cpu().byte())
        pil_img.save(os.path.join(out_dir, f"{prefix}_{step}_img_{i}.png"))

    torch.save(tokens.cpu(), os.path.join(out_dir, f"{prefix}_{step}_tokens.pt"))


def add_mask_to_plot(mask, img_size=256):
    root = int(mask.shape[-1] ** 0.5)
    multiplier = img_size // root
    offset = multiplier // 2

    it = 0
    for i in range(root):
        for j in range(root):
            plt.gca().add_patch(
                plt.Rectangle(
                    (i * multiplier, j * multiplier), multiplier, multiplier, fill=False, color="black", linewidth=1
                )
            )
            
            if mask[it] == 1:
                char = "■"
                color = "blue"
            elif mask[it] == 0:
                char = "x"
                color = "yellow"
            else:
                raise RuntimeError(f"This should not be in the mask: {mask[it]}")
            it += 1

            # Invert (i, j) for matplotlib
            plt.text(j * multiplier + offset - 3, i * multiplier + offset + 2, char, color=color, fontsize=10)
