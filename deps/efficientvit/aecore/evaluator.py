import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import MISSING
from torch.utils.data import DataLoader
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from tqdm import tqdm

from deps.efficientvit.ae_model_zoo import DCAE_HF, REGISTERED_DCAE_MODEL, MaskgitVqgan, MaskBit14Bit
from deps.efficientvit.aecore.data_provider.imagenet import ImageNetDataProvider, ImageNetDataProviderConfig
from deps.efficientvit.apps.metrics.fid.fid import FIDStats, FIDStatsConfig, WatermarkStats
from deps.efficientvit.apps.metrics.psnr.psnr import PSNRStats, PSNRStatsConfig
from deps.efficientvit.apps.utils.dist import (
    dist_barrier,
    dist_init,
    get_dist_local_rank,
    get_dist_rank,
    is_dist_initialized,
    is_master,
)
from deps.efficientvit.apps.utils.metric import AverageMeter
from deps.efficientvit.models.efficientvit.dc_ae import DCAE
from deps.efficientvit.models.utils.network import get_dtype_from_str, is_parallel
import copy

__all__ = ["EvaluatorConfig", "Evaluator"]


time_stamp = time.time()


def psnr_pair(img1, img2):
    mse = torch.mean((255 * (torch.tensor(img1) - torch.tensor(img2))) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(mse)


@dataclass
class EvaluatorConfig:
    run_dir: str = MISSING
    seed: int = 0

    evaluate_split: str = "test"
    evaluate_dir_name: Optional[str] = None
    num_save_images: int = 64
    save_images_at_all_procs: bool = False
    save_all_images: bool = False
    num_batches: int = -1  # number of batches to evaluate, -1 for all batches

    resolution: int = 256
    amp: str = "fp32"  # "bf16"

    # dataset
    dataset: str = "imagenet"
    imagenet: ImageNetDataProviderConfig = field(
        default_factory=lambda: ImageNetDataProviderConfig(resolution="${..resolution}")
    )

    # model
    model: str = MISSING

    # metrics
    compute_fid: bool = True
    fid: FIDStatsConfig = field(default_factory=FIDStatsConfig)
    compute_psnr: bool = True
    psnr: PSNRStatsConfig = field(default_factory=PSNRStatsConfig)
    compute_ssim: bool = True
    compute_lpips: bool = True


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg
        self.setup_dist_env()
        self.setup_seed()

        if cfg.amp == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            cfg.amp = "fp32"

        # data provider
        if cfg.dataset == "imagenet":
            self.data_provider = ImageNetDataProvider(cfg.imagenet)
        else:
            raise ValueError(f"dataset {cfg.dataset} is not supported")

        # model
        if cfg.model in REGISTERED_DCAE_MODEL:
            model = DCAE_HF.from_pretrained(f"mit-han-lab/{cfg.model}")
        elif cfg.model == "maskgit-vqgan":
            model = MaskgitVqgan()
        elif cfg.model == "maskbit":
            model = MaskBit14Bit()

        # Reference model and freeze its parameters.
        reference_network = copy.deepcopy(model)
        for param in reference_network.parameters():
            param.requires_grad = False
        reference_network.eval()
        self.reference_model = reference_network.cuda()

        # Freeze the encoder and disable its train mode.
        for name, param in model.named_parameters():
            if "encoder" in name or "quantizer" in name:
                param.requires_grad = False
            if "decoder" in name:
                param.requires_grad = True

        def disabled_train(self, mode=True):
            """Overwrite model.train with this function to make sure train/eval mode
            does not change anymore."""
            return self

        model.eval()
        model.encoder.train = disabled_train

        # if cfg.channels_last:
        #     model = model.to(memory_format=torch.channels_last)

        if is_dist_initialized():
            self.model = nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[get_dist_local_rank()], find_unused_parameters=True
            )
            self.rank = get_dist_rank()
        else:
            self.model = model.cuda()
            self.rank = 0

    def setup_dist_env(self) -> None:
        dist_init()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(get_dist_local_rank())

    def setup_seed(self) -> None:
        seed = get_dist_rank() + self.cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @property
    def enable_amp(self) -> bool:
        return self.cfg.amp != "fp32"

    @property
    def amp_dtype(self) -> torch.dtype:
        return get_dtype_from_str(self.cfg.amp)

    @property
    def network(self) -> DCAE:
        return self.model.module if is_parallel(self.model) else self.model
    
    @property
    def reference_network(self) -> DCAE:
        return self.reference_model.module if is_parallel(self.reference_model) else self.reference_model

    def run_step(self, images, global_step: int = 0):
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
            output, loss, info = self.model(images, global_step)
        return {"output": output, "loss": loss, "info": info}

    @torch.no_grad
    def evaluate_single_dataloader(
        self, dataloader: DataLoader, step: int, f_log=sys.stdout, additional_dir_name: str = ""
    ) -> dict[str, Any]:
        self.model.eval()
        valid_loss = AverageMeter(is_distributed=is_dist_initialized())
        valid_l1_loss_wm = AverageMeter(is_distributed=is_dist_initialized())
        valid_perceptual_loss_wm = AverageMeter(is_distributed=is_dist_initialized())
        valid_l1_loss_ori = AverageMeter(is_distributed=is_dist_initialized())
        valid_perceptual_loss_ori = AverageMeter(is_distributed=is_dist_initialized())
        valid_psnr_inmod_ori_ae = AverageMeter(is_distributed=is_dist_initialized())
        valid_psnr_posthoc_ori_ae = AverageMeter(is_distributed=is_dist_initialized())
        device = torch.device("cuda")

        # metrics
        compute_fid = self.cfg.compute_fid
        fid_stats = FIDStats(self.cfg.fid)
        if self.watermarker is not None:
            bit_acc_stats = WatermarkStats()
        if self.cfg.compute_psnr:
            psnr = PSNRStats(self.cfg.psnr)
        if self.cfg.compute_ssim:
            ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 255.0)).to(device)
        if self.cfg.compute_lpips:
            lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

        if self.cfg.evaluate_dir_name is not None:
            evaluate_dir = os.path.join(self.cfg.run_dir, self.cfg.evaluate_dir_name, additional_dir_name)
        else:
            evaluate_dir = os.path.join(self.cfg.run_dir, f"{step}", additional_dir_name)
        if is_master():
            os.makedirs(evaluate_dir, exist_ok=True)
        if is_dist_initialized():
            dist_barrier()

        with tqdm(
            total=len(dataloader),
            desc="Valid Steps #{}".format(step),
            disable=not is_master(),
            file=f_log,
            mininterval=10.0,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as t:
            num_saved_images = 0
            for idx, (images, _) in enumerate(dataloader):
                if idx == self.cfg.num_batches:
                    break
                # preprocessing
                images = images.cuda()
                # if self.cfg.channels_last:
                #     images = images.to(memory_format=torch.channels_last)
                # forward
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                    x_wm_inmodel, loss, _ = self.model(images, global_step=0)
                    x_wm_posthoc, _, _ = self.reference_network(images, global_step=0, watermarker=self.watermarker, msg=self.msg)
                    x_non_wm, _, _ = self.reference_network(images, global_step=0)

                    # Losses
                    l1_loss_wm = torch.abs(x_wm_inmodel.contiguous() - x_wm_posthoc.contiguous()).mean()
                    perceptual_loss_wm = self.lpips(x_wm_posthoc.contiguous(), x_wm_inmodel.contiguous()).mean()
                    l1_loss_ori = torch.abs(x_wm_inmodel.contiguous() - x_non_wm.contiguous()).mean()
                    perceptual_loss_ori = self.lpips(x_non_wm.contiguous(), x_wm_inmodel.contiguous()).mean()

                input_images = images * 0.5 + 0.5
                ori_ae_images = (x_non_wm * 0.5 + 0.5).clamp(0, 1)
                output_images = (x_wm_inmodel * 0.5 + 0.5).clamp(0, 1)
                wm_posthoc_images = (x_wm_posthoc * 0.5 + 0.5).clamp(0, 1)
                if (
                    num_saved_images < self.cfg.num_save_images and (is_master() or self.cfg.save_images_at_all_procs)
                ) or self.cfg.save_all_images:
                    device = images.device
                    
                    for j in range(input_images.shape[0]):
                        # save_image(
                        #     torch.cat([input_images[j : j + 1], output_images[j : j + 1]], dim=3),
                        #     os.path.join(evaluate_dir, f"{self.rank}_{num_saved_images}.png"),
                        # )
                        save_image(
                            input_images[j : j + 1],
                            os.path.join(evaluate_dir, f"{self.rank}_{num_saved_images}_ori.png"),
                        )
                        save_image(
                            ori_ae_images[j : j + 1],
                            os.path.join(evaluate_dir, f"{self.rank}_{num_saved_images}_ori_ae.png"),
                        )
                        save_image(
                            output_images[j : j + 1],
                            os.path.join(evaluate_dir, f"{self.rank}_{num_saved_images}_wm_inmodel.png"),
                        )
                        save_image(
                            wm_posthoc_images[j : j + 1],
                            os.path.join(evaluate_dir, f"{self.rank}_{num_saved_images}_wm_posthoc.png"),
                        )
                        save_image(
                            (wm_posthoc_images[j : j + 1] - ori_ae_images[j : j + 1]).mean(1, keepdim=True) * 0.5 + 0.5,
                            os.path.join(evaluate_dir, f"{self.rank}_{num_saved_images}_diff_posthoc_ori_ae.png"),
                        )
                        save_image(
                            (output_images[j : j + 1] - ori_ae_images[j : j + 1]).mean(1, keepdim=True) * 0.5 + 0.5,
                            os.path.join(evaluate_dir, f"{self.rank}_{num_saved_images}_diff_inmodel_ori_ae.png"),
                        )
                        save_image(
                            (wm_posthoc_images[j : j + 1] - output_images[j : j + 1]).mean(1, keepdim=True) * 0.5 + 0.5,
                            os.path.join(evaluate_dir, f"{self.rank}_{num_saved_images}_diff_posthoc_inmodel.png"),
                        )
                        num_saved_images += 1
                        if num_saved_images >= self.cfg.num_save_images and not self.cfg.save_all_images:
                            break

                # update metrics
                valid_loss.update(loss.detach().item(), 1)
                valid_l1_loss_wm.update(l1_loss_wm.detach().item(), 1)
                valid_perceptual_loss_wm.update(perceptual_loss_wm.detach().item(), 1)
                valid_l1_loss_ori.update(l1_loss_ori.detach().item(), 1)
                valid_perceptual_loss_ori.update(perceptual_loss_ori.detach().item(), 1)
                valid_psnr_inmod_ori_ae.update(psnr_pair(output_images, ori_ae_images), 1)
                valid_psnr_posthoc_ori_ae.update(psnr_pair(wm_posthoc_images, ori_ae_images), 1)
                if compute_fid:
                    device = x_wm_inmodel.device
                    fid_stats.add_data(output_images)
                images_ref_uint8 = (255 * input_images + 0.5).clamp(0, 255).to(torch.uint8)
                images_pred_uint8 = (255 * output_images + 0.5).clamp(0, 255).to(torch.uint8)
                if self.watermarker is not None:
                    watermark_logits = self.watermarker.detect(images_pred_uint8.float() / 255., is_video=False)["preds"][:, 1:]
                    pred_bits = (watermark_logits > 0).float()
                    bit_acc = (pred_bits == self.msg).cpu().numpy().astype(np.float32).mean(1)
                    bit_acc_stats.add_data(bit_acc)
                if self.cfg.compute_psnr:
                    psnr.add_data(images_ref_uint8, images_pred_uint8)
                if self.cfg.compute_ssim:
                    ssim.update(images_ref_uint8, images_pred_uint8)
                if self.cfg.compute_lpips:
                    lpips.update(images_ref_uint8 / 255, images_pred_uint8 / 255)
                ## tqdm
                postfix_dict = {
                    "loss": valid_loss.avg,
                    "bs": images.shape[0],
                    "res": images.shape[2],
                }
                t.set_postfix(postfix_dict, refresh=False)
                t.update()
        valid_info_dict = {
            "valid_loss": valid_loss.avg,
            "valid_l1_loss_wm": valid_l1_loss_wm.avg,
            "valid_perceptual_loss_wm": valid_perceptual_loss_wm.avg,
            "valid_l1_loss_ori": valid_l1_loss_ori.avg,
            "valid_perceptual_loss_ori": valid_perceptual_loss_ori.avg,
            "valid_psnr_inmod_ori_ae": valid_psnr_inmod_ori_ae.avg,
            "valid_psnr_posthoc_ori_ae": valid_psnr_posthoc_ori_ae.avg,
        }
        torch.cuda.empty_cache()
        if compute_fid:
            valid_info_dict["valid_fid"] = fid_stats.compute_fid()
        if self.watermarker is not None:
            valid_info_dict["valid_mean_bit_acc"] = bit_acc_stats.get_mean_bit_acc()
        if self.cfg.compute_psnr:
            valid_info_dict["valid_psnr_inmod_ori"] = psnr.compute()
        if self.cfg.compute_ssim:
            valid_info_dict["valid_ssim"] = ssim.compute().item()
        if self.cfg.compute_lpips:
            valid_info_dict["valid_lpips"] = lpips.compute().item()
        return valid_info_dict

    @torch.no_grad
    def evaluate(self, step: int, f_log=sys.stdout) -> dict[str, Any]:
        if self.cfg.evaluate_split == "train":
            dataloader = self.data_provider.train
        elif self.cfg.evaluate_split == "valid":
            dataloader = self.data_provider.valid
        elif self.cfg.evaluate_split == "test":
            dataloader = self.data_provider.test
        else:
            raise NotImplementedError
        valid_info_dict = self.evaluate_single_dataloader(dataloader, step, f_log)
        return valid_info_dict
