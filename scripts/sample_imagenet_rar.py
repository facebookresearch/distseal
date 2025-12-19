# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Sampling scripts for RAR-XL on ImageNet.

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

Reference: 
    https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
"""
"""
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --rdzv-endpoint=localhost:9999 sample_imagenet_rar.py \
    config=configs/generation/rar.yaml \
    experiment.output_dir="rar_xl" \
    experiment.generator_card=cards/rar_xl_distseal.yaml \
    experiment.decoder_card=cards/maskgit_base.yaml \
    model.generator.randomize_temperature=1.0 \
    model.generator.guidance_scale=16.0 \
    model.generator.guidance_scale_pow=2.75
"""
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import os
from omegaconf import OmegaConf
from tqdm import tqdm

from distseal.loader import load_decoder, load_generator
from rar.utils.generation_utils import sample_fn
from rar.utils.misc import is_main_process


def get_config_cli():
    cli_conf = OmegaConf.from_cli()
    
    try:
         yaml_conf = OmegaConf.load(cli_conf.config)
    except:
        print(f"Failed to load config file from {cli_conf.config}")
        raise
    return OmegaConf.merge(yaml_conf, cli_conf)


def main():
    config = get_config_cli()
    num_cond = int(config.model.generator.condition_num_classes)
    samples_per_class = config.model.generator.get("samples_per_class", 50)
    num_fid_samples = num_cond * samples_per_class
    per_proc_batch_size = 50
    sample_folder_dir = config.experiment.output_dir    
    seed = 42

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    # setup DDP.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}")
    )

    world_size = dist.get_world_size() 
    device = local_rank  # Use integer, not torch.device
    seed = seed + local_rank
    torch.manual_seed(seed)
    print(f"Starting rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    if is_main_process():
        print(f"Config:\n{OmegaConf.to_yaml(config)}")
        if config.model.vq_model.get("name", None):
            print(f"Using VQ model: {config.model.vq_model.name}")

    try:
        # Load models sequentially across ranks to avoid OOM from simultaneous CPU loads
        for rank_to_load in range(world_size):
            if local_rank == rank_to_load:
                tokenizer_card = config.experiment.get("decoder_card", "maskgit_base.yaml")
                print(f"[Rank {local_rank}] Loading decoder from {tokenizer_card}...")
                tokenizer = load_decoder(tokenizer_card, device=device)
                
                generator_card = config.experiment.get("generator_card", "rar_xl_base.yaml")
                print(f"[Rank {local_rank}] Loading generator from {generator_card}...")
                generator = load_generator(generator_card, device=device)
                
                print(f"[Rank {local_rank}] Successfully loaded tokenizer and generator.")
            dist.barrier()
    except Exception as e:
        print(f"[Rank {local_rank}] FAILED to load models: {e}")
        import traceback
        traceback.print_exc()
        raise

    if is_main_process():
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    assert num_fid_samples % global_batch_size == 0
    if is_main_process():
        print(f"Total number of images that will be sampled: {num_fid_samples}")

    samples_needed_this_gpu = int(num_fid_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if is_main_process() else pbar
    total = 0


    num_cond = int(config.model.generator.condition_num_classes)
    all_classes = list(range(num_cond)) * samples_per_class

    if world_size > 1:
        subset_len = len(all_classes) // world_size
        all_classes = np.array(all_classes[local_rank * subset_len: (local_rank+1)*subset_len], dtype=np.int64)
    else:
        all_classes = np.array(all_classes, dtype=np.int64)
    cur_idx = 0

    for _ in pbar:
        y = torch.from_numpy(all_classes[cur_idx * n: (cur_idx+1)*n]).to(device)
        cur_idx += 1

        samples = sample_fn(
            generator=generator,
            tokenizer=tokenizer,
            labels=y.long(),
            randomize_temperature=config.model.generator.randomize_temperature,
            guidance_scale=config.model.generator.guidance_scale,
            guidance_scale_pow=config.model.generator.guidance_scale_pow,
            device=device
        )
        
        # Save samples to disk as individual .png files.
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + local_rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    # dist.barrier()
    # if rank == 0:
    #     create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
    #     print("Done.")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
