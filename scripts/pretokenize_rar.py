# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Pretokenization script for TiTok and RAR, with optional watermarking.

Reference:
    https://github.com/bytedance/1d-tokenizer/blob/main/scripts/pretokenization.py
    
Copyright Notice:

===================================================================================
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
===================================================================================


Example command:

PYTHONPATH=deps torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --rdzv-endpoint=localhost:9999 \
    scripts/pretokenize_rar.py \
    --img_size 256 \
    --batch_size 8 \
    --ten_crop \
    --data_path ${PATH_TO_IMAGENET} \
    --cached_path ${PATH_TO_SAVE_JSONL} \
    --wm_path ${PATH_TO_WATERMARK_MODEL} \
    --scaling_w 1.5 \
    --wm_message ${PATH_TO_SAVE_WATERMARK_MESSAGE_FILE}
"""
import argparse
import datetime
from pathlib import Path
import numpy as np

import os
import time

import torch
from tqdm import tqdm
import json

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms

from rar.data.local_reader import ImageFolderWithFilename, ImageTransform
from rar.data.preprocess import center_crop_arr
import rar.utils.misc as misc

from distseal.loader import load_decoder


class CachedHiddenStatesDataset(torch.utils.data.Dataset):
    def __init__(self, cached_path):
        self.cached_path = cached_path
        self.data = self.load_cached_data()

    def load_cached_data(self):
        # Load the cached data from the specified path
        if os.path.exists(self.cached_path):
            with open(self.cached_path, 'rb') as f:
                return torch.load(f)
        return []

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_args_parser():
    parser = argparse.ArgumentParser('Cache VQ codes', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--num_samples', default=0, type=int,
                        help='number of samples to cache, if None, cache all samples')
    
    # tokenizer parameters
    parser.add_argument('--tokenizer', default='', type=str,
                        help='path to the tokenizer model used to pretokenize the images')
    # Watermark parameters
    parser.add_argument('--wm_path', default='', type=str,
                        help='path to the watermarker model')
    parser.add_argument('--scaling_w', type=float, default=0.7,
                        help='scaling factor for the watermarked tokens')
    parser.add_argument('--wm_message', type=str, default='',
                        help='watermark message to embed in the tokens. This can be a string of a path to a file we will save the randomed message to.')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--split", type=str, default="train", help="which data split to pretokenize")

    # caching latents
    parser.add_argument('--cached_path', default='', help='path to cache the non-watermarked tokenization')
    
    parser.add_argument('--output_path', default='', help='path to output the watermarked tokenization')

    parser.add_argument("--ten_crop", action='store_true', help="whether using random crop")

    parser.add_argument("--flip", action='store_true', help="whether using horizontal flip")

    parser.add_argument('--random_crop', action='store_true', help="whether using random cropping")
    return parser

@torch.no_grad()
def main(args):
    os.makedirs(args.cached_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.ten_crop:
        # augmentation following LLamaGen
        crop_size = int(args.img_size * 1.1)
        transform_ops = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.img_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
        ])
    elif args.random_crop:
        print("Using random cropping")
        transform_ops = ImageTransform(
            resize_shorter_edge=256,
            crop_size=256,
            random_crop=False,
            random_flip=True,
            normalize_mean=[0., 0., 0.],  # Maskgit-VQ expects images in [0, 1] range
            normalize_std=[1., 1., 1.],
        ).train_transform
    else:
        # augmentation following DiT and ADM
        transform_ops = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # MaskGIT-VQ expects input in range of [0, 1]
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    num_samples = args.num_samples or None
    
    dataset = ImageFolderWithFilename(os.path.join(args.data_path, args.split), transform=transform_ops, num_samples=num_samples)

    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler = %s" % str(sampler))

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Don't drop in cache
    )
    
    # Download MaskGit-VQ tokenizer weights if not present
    if global_rank == 0:
        if not Path("maskgit-vqgan-imagenet-f16-256.bin").exists():
            print("Downloading MaskGit-VQ tokenizer weights...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="fun-research/TiTok",
                filename="maskgit-vqgan-imagenet-f16-256.bin", local_dir="./")

    if misc.is_dist_avail_and_initialized():
        torch.cuda.synchronize()

    tokenizer = load_decoder(args.tokenizer)
    tokenizer.to(device)

    # Set up the watermarking
    if args.wm_path:
        from distseal.utils.cfg import setup_model_from_checkpoint
        wm_model = setup_model_from_checkpoint(args.wm_path)
        wm_model.eval()
        wm_model.to(device)
    else:
        wm_model = None

    scaling_w = args.scaling_w

    if args.wm_message:
        if Path(args.wm_message).exists():
            with open(args.wm_message, 'r', encoding="utf-8") as f:
                msg_tensor = misc.message_to_tensor(f.read().strip())
        else:
            # wm_message can be a string or a path to a file
            assert wm_model is not None, "wm_message can only be used with a watermarking model"
            try:
                msg_tensor = misc.message_to_tensor(args.wm_message.strip())
            except Exception:
                msg_tensor = wm_model.get_random_msg(1).to("cuda")
                with open(args.wm_message, 'w') as f:
                    f.write(misc.tensor_to_message(msg_tensor))

        msg_tensor = msg_tensor.to(device)
    else:
        msg_tensor = None

    processed = []
    total_processed = 0
    # hidden_states_processed = []
    wm_processed = []
    
    buffer_size = 10_000
    idx = 0

    print(f"Start caching latents, {args.rank}, {args.gpu}")
    target_path = f"{args.cached_path}/pretokenized_{args.rank}"
    
    if global_rank == 0:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    dist.barrier()
    target_json_path = target_path + ".jsonl"

    wm_target_path = f"{args.output_path}/wm_pretokenized_{args.rank}"
    wm_target_json_path = wm_target_path + ".jsonl"

    if Path(target_json_path).exists():
        print(f"Rank {args.rank} found existing cached file {target_json_path}, skipping caching.")
        return
    
    start_time = time.time()
    for samples, target, _ in tqdm(data_loader):
        try:
            samples = samples.to(device, non_blocking=True)    
            if args.ten_crop:
                samples_all = samples.flatten(0, 1)
                target_all = target.unsqueeze(1).repeat(1, 10).flatten(0, 1)
            elif args.flip:
                samples_all = torch.cat([samples, torch.flip(samples, dims=[-1])])
                target_all = torch.cat([target, target])
            else:
                samples_all = samples
                target_all = target

            with torch.no_grad():
                hidden_states = tokenizer.encoder(samples_all)
                _, codebook_indices, _ = tokenizer.quantize(hidden_states)
                codes = codebook_indices.detach()
                codes_wm = None

                if wm_model is not None and msg_tensor is not None:
                    # Repeat message tensor for current batch size
                    batch_msg_tensor = msg_tensor.repeat(samples_all.shape[0], 1)
                    hidden_states_wm = wm_model.embedder(hidden_states, msgs=batch_msg_tensor)
                    hidden_states = hidden_states + scaling_w * hidden_states_wm

                    _, codebook_indices, _ = tokenizer.quantize(hidden_states)
                    codes_wm = codebook_indices.detach()
        
        except Exception as e:
            # Debug: Check input shapes
            print(f"Samples shape: {samples.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Target values: min={target.min()}, max={target.max()}")
            
            print(f"Processed samples shape: {samples_all.shape}")
            print(f"Processed target shape: {target_all.shape}")
            print(f"Error occurred: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            break
    
        hidden_states = hidden_states.cpu()
        for b in range(codes.shape[0]):
            processed.append({
                "class_id": target_all[b].cpu().item(),
                "tokens": codes[b].tolist()
            })
            if codes_wm is not None:
                wm_processed.append({
                    "class_id": target_all[b].cpu().item(),
                    "tokens": codes_wm[b].tolist()
                })
            # hidden_states_processed.append(hidden_states[b])
            
        if len(processed) >= buffer_size:
            # if misc.is_dist_avail_and_initialized():
            #     torch.cuda.synchronize()
            # print(f"{args.rank} proccessed {len(processed)} samples, saving to {target_json_path}")
            # Save the processed data to a JSON file
            with open(target_json_path, "a") as json_f:
                for entry in processed:
                    json_f.write(json.dumps(entry) + "\n")
            # torch.save(hidden_states_processed, f"{target_path}/{idx}.pth")
            idx += 1
            total_processed += len(processed)
            processed = []
            # hidden_states_processed = []

            if wm_processed:
                with open(wm_target_json_path, "a") as json_f:
                    for entry in wm_processed:
                        json_f.write(json.dumps(entry) + "\n")
                wm_processed = []

    if misc.is_dist_avail_and_initialized():
        torch.cuda.synchronize()

    if len(processed) > 0:
        print(f"{args.rank} proccessed {len(processed)} samples, saving to {target_json_path}")
        # Save the processed data to a JSON file
        with open(target_json_path, "a") as json_f:
            for entry in processed:
                json_f.write(json.dumps(entry) + "\n")
        # torch.save(hidden_states_processed, f"{target_path}/{idx}.pth")
        idx += 1
        total_processed += len(processed)
        processed = []
        # hidden_states_processed = []

        if wm_processed:
            with open(wm_target_json_path, "a") as json_f:
                for entry in wm_processed:
                    json_f.write(json.dumps(entry) + "\n")
            wm_processed = []

    print(f"[Rank {args.rank}] proccessed {total_processed} samples")
    
    if misc.is_dist_avail_and_initialized():
        torch.cuda.synchronize()

    # write into a single jsonl
    if global_rank == 0:
        misc.convert_json_to_jsonl(f"{args.cached_path}/pretokenized_*.json",
                              f"{args.cached_path}/pretokenized.jsonl")

    if misc.is_dist_avail_and_initialized():
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
