# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path

import warnings

from distseal.utils.checkpoint_manager import resolve_checkpoint
from distseal.utils.dist import is_main_process
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.distributed as dist

from typing import Dict, Optional

from omnisealbench import task, get_model
from omnisealbench.utils.detection import get_detection_and_decoded_keys
from omnisealbench.utils.analysis import aggregate_by_attack_variants

from distseal.loader import load_detector, load_message, parse_model_card

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*load_state_dict.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant parameter should be passed explicitly.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*None of the inputs have requires_grad=True.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")

torch.set_grad_enabled(False)
device = torch.device("cuda")


class LatentWatermark:
    """Wrapper of DistSeal latent watermark model for Omnisealbench evaluation."""
    
    model: nn.Module
    
    def __init__(self, model: torch.nn.Module, img_size: float = 256, nbits: int = 64, detection_bits: int = 0):

        self.model = model
            
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
        ])        
        
        # Each model should have an attribute 'nbits'. If the model does not have this attribute,
        # we must set the value `message_size` in the task. If Omniseal could not find information
        # from either model or the task, it will raise the ValueError
        self.nbits = nbits
        self.detection_bits = detection_bits
    
    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: torch.Tensor,
        detection_threshold: float = 0.0,
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        image_tensors = []
        # for img in contents:
        #     img_tensor = self.transform(img).unsqueeze(0).to(device)
        #     image_tensors.append(img_tensor)
        # image_tensors = torch.cat(image_tensors, dim=0)
        image_tensors = self.transform(contents).to(device)
        extracted_bits = self.model(image_tensors)
        extracted_bits = extracted_bits[:, 1:]
        exp = get_detection_and_decoded_keys(
            extracted_bits,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold,
        )
        return exp

def build_latent_watermark_model(detector_card: str, img_size: int = 512, nbits: int = 64, detection_bits: int = 0, device: str = "cpu") -> LatentWatermark:        
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    try:
        # Load models sequentially across ranks to avoid OOM from simultaneous CPU loads
        for rank_to_load in range(world_size):
            rank = dist.get_rank()
            if rank == rank_to_load:
                print(f"[Rank {rank}] Loading detector from {detector_card}...")
                watermarker, _ = load_detector(detector_card)    
                watermarker = watermarker.eval()
                watermarker = watermarker.to(device)
            dist.barrier()
    except Exception as e:
        print(f"[Rank {rank}] FAILED to load models: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return LatentWatermark(model=watermarker, img_size=img_size, nbits=nbits, detection_bits=detection_bits)


def run_eval(args):
    image_folder = args.image_dir
    result_dir = args.result_dir
    detector_card = args.detector_card
    batch_size = args.batch_size
    resolution = args.resolution
    attacks = args.attacks
    num_samples = args.num_samples
    
    conf = parse_model_card(detector_card)
    msg_file = resolve_checkpoint(conf.message)
    msg_tensor = load_message(msg_file)
    
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
        
    detection_task = task(
        "detection",
        modality="image",
        seed=args.seed,
        dataset_dir=str(image_folder),
        result_dir=str(Path(result_dir) / "attacks"),
        original_image_pattern=None,
        watermarked_image_pattern="**/*.png",
        message_pattern=None,
        even_shapes=True,
        metrics=None,  # only calculate bit accuracy
        attacks=attacks,
        batch_size=batch_size,
        detection_bits=0,
        num_samples=num_samples,
    )
    builder_args = {"as_type": "detector", "detector_card": detector_card, "img_size": resolution, "nbits": 64, "device": "cuda"}

    detector = get_model(build_latent_watermark_model, **builder_args)
    
    # move to correct device of the process
    msg_tensor = msg_tensor.to("cuda")
    _, raw_results = detection_task(detector, wm_message=msg_tensor, auto_distributed=False)
    
    if is_main_process():
        results = detection_task.print_scores(raw_results)
        results = aggregate_by_attack_variants(results)

        grouped_summary = results.groupby(['attack']).agg({'bit_acc': 'mean', 'log10_p_value': 'mean'}).reset_index()
        grouped_details = results.groupby(['attack', 'attack_variant']).agg({'bit_acc': 'mean', 'log10_p_value': 'mean'}).reset_index()

        # Save result_df to a file in result_dir if specified
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, "results_summary.csv")
            grouped_summary.to_csv(result_path, index=False)
            result_path = os.path.join(result_dir, "results_details.csv")
            grouped_details.to_csv(result_path, index=False)

        print(grouped_summary.to_string())
    
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run omnisealbench eval script.")
    parser.add_argument("--image_dir", type=str, help="Parent directory containing subfolders of images.")
    parser.add_argument("--result_dir", type=str, help="Directory where the results of the watermark generation will be saved.", default=None)
    parser.add_argument("--log_dir", type=str, help="Directory where the (SLURM job) logs will be saved.", default="/checkpoint/avseal/tuantran/logs/slurm/latent/rar")
    parser.add_argument("--detector_card", type=str, help="Path to the detector card file.")
    parser.add_argument("--batch_size", type=int, help="Latent watermark decoder batch size.", default=32)
    parser.add_argument("--resolution", type=int, help="Resolution to resize images for evaluation.", default=256)
    parser.add_argument("--attacks", type=str, help="Path to the attacks configuration file.")
    parser.add_argument("--num_samples", type=int, help="Number of samples to use for evaluation.", default=1)
    parser.add_argument("--job_name", type=str, help="Job name for slurm launcher.", default="omnisealbench_eval_submitit")
    parser.add_argument("--model_type", type=str, help="model type (latent or baseline)", default="latent")
    parser.add_argument("--seed", type=int, help="seed for reproducibility", default=42)
        
    args = parser.parse_args()
    run_eval(args)