# Copyright (c) Meta Platforms, Inc. and affiliates.

# Base transformer
#
import torch
from argparse import ArgumentParser


# TO-BE-REMOVED-COMMENT:
# from efficientvit.diffusion_model_zoo import DCAE_Diffusion_HF
#
# torch.save(new_state_dict, "/checkpoint/avseal/models/distseal/dcae/uvit-h-in-512px-train2000k.pt")
# Distilled transformer
#ckpt = torch.load("/checkpoint/avseal/sylvestre/2025_logs/0923_dcae_diffusion_distillation/_reference_model_weight=0.0/checkpoint.pt", weights_only=False, map_location="cpu")



def extract_uvit(ckpt_path, save_path, ema_weight=0.9999):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    new_state_dict = {}
    for k, v in ckpt['ema'][ema_weight].items():
        if k.startswith("uvit."):
            new_state_dict[k] = v
    torch.save(new_state_dict, save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint file of the distilled DCAE diffusion model")
    parser.add_argument("--save_path", type=str, help="Path to save the extracted UVIT weights")
    parser.add_argument("--ema_weight", type=float, default=0.9999, help="EMA weight to extract from the checkpoint")
    args = parser.parse_args()
    
    extract_uvit(args.ckpt, args.save_path, ema_weight=args.ema_weight)
