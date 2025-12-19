# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Example usage (cluster 2 gpus):
    torchrun --nproc_per_node=2 train.py --local_rank 0
Example usage (cluster 1 gpu):
    torchrun train.py --debug_slurm
    For eval full only:
        torchrun train.py --debug_slurm --only_eval True --output_dir output/

Put OMP_NUM_THREADS such that OMP_NUM_THREADS=(number of CPU threads)/(nproc per node) to remove warning messages
        
Examples:
    OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 train.py --local_rank 0 \
        --image_dataset sa-1b-full-resized --workers 8 \
        --extractor_model convnext_tiny --embedder_model unet_small2_yuv_quant --hidden_size_multiplier 1 --nbits 128 \
        --scaling_w_schedule Cosine,scaling_min=0.2,start_epoch=200,epochs=200 --scaling_w 1.0 --scaling_i 1.0 --attenuation jnd_1_1 \
        --epochs 601 --iter_per_epoch 1000 --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=601,warmup_lr_init=1e-8,warmup_t=20 --optimizer AdamW,lr=5e-4 \
        --lambda_dec 1.0 --lambda_d 0.1 --lambda_i 0.1 --perceptual_loss yuv  --num_augs 2 --augmentation_config configs/all_augs_v3.yaml --disc_in_channels 1 --disc_start 50 
  
"""

import argparse
import datetime
import json
import os
import time
from typing import List, Optional

import numpy as np
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision.utils import save_image

import distseal.utils as utils
import distseal.utils.dist as udist
import distseal.utils.logger as ulogger
import distseal.utils.optim as uoptim
from distseal.augmentation import (get_validation_augs,
                                    get_validation_augs_subset)
from distseal.augmentation.augmenter import Augmenter
import distseal.augmentation.neuralcompression as neuralcompression
from distseal.data.loader import get_dataloader_segmentation
from distseal.data.transforms import get_resize_transform
from distseal.utils.metrics import bit_accuracy, psnr, ssim
from distseal.losses.detperceptual import DetectionLoss
from distseal.models import VideoWam, Wam, build_embedder, build_extractor
from distseal.modules.jnd import JND, VarianceBasedJND
from distseal.utils.data import parse_dataset_params
from distseal.utils.image import create_diff_img
from distseal.utils.tensorboard import CustomTensorboardWriter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


import argparse
import sys
from omegaconf import OmegaConf


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)
    
    # # the first positional argument is the config file
    group = parser.add_argument_group('Config file')
    # parser.add_argument("config", default=None, help="Path to YAML config file")
    aa("--config", type=str, 
        help="Path to YAML config file", default=None)

    group = parser.add_argument_group('Dataset parameters')
    aa("--dataset.train_dir", type=str, 
        help="Path to the train split of the image dataset.", default="")
    aa("--dataset.val_dir", type=str, 
        help="Path to the validation split of the image dataset.", default="")
    aa("--dataset.train_annotation_file", type=str, 
        help="Path to the annotation file for the train split.", default="")
    aa("--dataset.val_annotation_file", type=str, 
        help="Path to the annotation file for the validation split.", default="")

    group = parser.add_argument_group('Experiments parameters')
    aa("--output_dir", type=str, default="output/",
       help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Embedder and extractor config')
    aa("--embedder_config", type=str, default="configs/embedder.yaml",
       help="Path to the embedder config file")
    aa("--extractor_config", type=str, default="configs/extractor.yaml",
       help="Path to the extractor config file")
    aa("--attenuation_config", type=str, default="configs/attenuation.yaml",
       help="Path to the attenuation config file")
    aa("--embedder_model", type=str, default="unet_small2_yuv_quant",
       help="Name of the extractor model")
    aa("--extractor_model", type=str, default="convnext_tiny",
       help="Name of the extractor model")

    group = parser.add_argument_group('Augmentation parameters')
    aa("--augmentation_config", type=str, default="configs/augmentation/all_augs_v3.yaml",
       help="Path to the augmentation config file")
    aa("--num_augs", type=int, default=2,
       help="Number of augmentations to apply")

    group = parser.add_argument_group('Image and watermark parameters')
    aa("--nbits", type=int, default=64,
       help="Number of bits used to generate the message. If 0, no message is used.")
    aa("--hidden_size_multiplier", type=float, default=1,
         help="Hidden size multiplier for the message processor")
    aa("--img_size", type=int, default=256,
       help="Size of the input images for data preprocessing, used at loading time for training.")
    aa("--img_size_val", type=int, default=256,
       help="Size of the input images for data preprocessing, used at loading time for validation.")
    aa("--img_size_proc", type=int, default=256, 
       help="Size of the input images for interpolation in the embedder/extractor models")
    aa("--resize_only", type=utils.bool_inst, default=False,
         help="If True, only resize the image no crop is applied at loading time (without preserving aspect ratio)")
    aa("--attenuation", type=str, default="None", help="Attenuation model to use")
    aa("--blending_method", type=str, default="additive",
       help="The blending method to use. Options include: additive, multiplicative ..etc see Blender Class for more")
    aa("--scaling_w", type=float, default=0.2,
       help="Scaling factor for the watermark in the embedder model")
    aa("--scaling_w_schedule", type=str, default="Cosine,scaling_min=0.02,start_epoch=100,epochs=100",
       help="Scaling factor for the watermark in the embedder model. Ex: 'Linear,scaling_min=0.025,epochs=100,start_epoch=0'")
    aa("--scaling_i", type=float, default=1.0,
       help="Scaling factor for the image in the embedder model")
    # VideoWam parameters related how to do video watermarking inference
    aa("--videowam_chunk_size", type=int, default=32,
       help="The number of frames to encode at a time.")
    aa("--videowam_step_size", type=int, default=4,
       help="The number of frames to propagate the watermark to.")
    aa("--lowres_attenuation", type=utils.bool_inst, default=False,
       help="Apply attenuation at low resolution for high-res images (more memory efficient)")
    aa("--autoencoder", type=str, default="None",
       help="Autoencoder model when watermarking in the latent space.")
    aa("--latent_layer", type=str, default="input",
       help="Watermarking layer when watermarking in the latent space.")
    aa("--latent_layer_watermarker_input", type=utils.bool_inst, default=True,
       help="Watermarker taking the watermarked layer as input.")
    aa("--latent_layer_normalization", type=utils.bool_inst, default=False,
       help="Normalizing the latent or not.")
    aa("--ae_quantize", type=utils.bool_inst, default=True,
       help="Using the quantization in the autoencoder model.")

    group = parser.add_argument_group('Optimizer parameters')
    aa("--optimizer", type=str, default="AdamW,lr=5e-4",
       help="Optimizer (default: AdamW,lr=5e-4)")
    aa("--optimizer_d", type=str, default=None,
       help="Discriminator optimizer. If None uses the same params (default: None)")
    aa("--scheduler", type=str, default="CosineLRScheduler,lr_min=1e-6,t_initial=601,warmup_lr_init=1e-8,warmup_t=20",
       help="Scheduler (default: None)")
    aa('--epochs', default=601, type=int,
       help='Number of total epochs to run')
    aa('--iter_per_epoch', default=1000, type=int,
       help='Number of iterations per epoch, made for very large datasets')
    aa('--sleepwake', type=utils.bool_inst, default=False,
       help='If True and lambda_d > 0 then do epoch optimize 0 and epoch optimizer 1 otherwise optimize them simultaneously')
    aa('--iter_per_valid', default=10, type=int,
       help='Number of iterations per eval, made for very large eval datasets if None eval on all dataset')
    aa('--resume_from', default=None, type=str,
       help='Path to the checkpoint to resume from')
    aa('--resume_disc', type=utils.bool_inst, default=False,
       help='If True, also load discriminator weights when resuming from checkpoint')
    aa('--resume_optimizer_state', type=utils.bool_inst, default=False,
       help='If True, also load optimizer state when resuming from checkpoint')
    aa("--finetune_detector_start", type=int, default=1e6,
       help="Number of epochs afterwhich the generator is frozen and detector is finetuned")

    group = parser.add_argument_group('Losses parameters')
    aa('--temperature', default=1.0, type=float,
       help='Temperature for the mask loss')
    aa('--lambda_det', default=0.0, type=float,
       help='Weight for the watermark detection loss')
    aa('--lambda_dec', default=1.0, type=float,
       help='Weight for the watermark decoding loss')
    aa('--lambda_i', default=0.0, type=float, help='Weight for the image loss')
    aa('--lambda_d', default=0.1, type=float,
       help='Weight for the discriminator loss')
    aa('--balanced', type=utils.bool_inst, default=False,
       help='If True, the weights of the losses are balanced')
    aa('--total_gnorm', default=1.0, type=float,
       help='Total norm for the adaptive weights. If 0, uses the norm of the biggest weight.')
    aa('--perceptual_loss', default='mse', type=str,
       help='Perceptual loss to use. "lpips", "watson_vgg" or "watson_fft"')
    aa('--disc_start', default=200, type=float,
       help='When does the discriminator loss start')
    aa('--disc_num_layers', default=3, type=int,
       help='Number of layers for the discriminator')
    aa('--maskbit_disc', type=utils.bool_inst, default=True,
       help='If True, the discriminator uses the maskbit architecture.')
    aa('--maskbit_maxpool', default=8, type=int,
       help='Size of the max pooling layer for the maskbit discriminator')
    aa('--lecam_regularization_weight', type=float, default=0.0,
       help='Weight for the LeCam regularization loss')
    aa('--disc_in_channels', default=3, type=int,
         help='Number of input channels for the discriminator')
    aa('--discriminator_on_autoencoded', type=utils.bool_inst, default=True,
       help='If True, the discriminator is applied to the autoencoded images')
    aa('--disc_scales', type=int, default=1,
       help='Number of scales for the discriminator')

    group = parser.add_argument_group('Loading parameters')
    aa('--batch_size', default=32, type=int, help='Batch size')
    aa('--batch_size_eval', default=8, type=int, help='Batch size for evaluation')
    aa('--workers', default=8, type=int, help='Number of data loading workers')

    group = parser.add_argument_group('Misc.')
    aa('--only_eval', type=utils.bool_inst,
       default=False, help='If True, only runs evaluate')
    aa('--eval_freq', default=10, type=int, help='Frequency for evaluation')
    aa('--full_eval_freq', default=50, type=int,
       help='Frequency for full evaluation')
    aa('--saveimg_freq', default=50, type=int, help='Frequency for saving images')
    aa('--saveckpt_freq', default=50, type=int, help='Frequency for saving ckpts')
    aa('--seed', default=0, type=int, help='Random seed')

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)

    return parser


def parse_args(parser):
    default_params = parser.parse_args([])
    cli_params = parser.parse_args()
       
    if cli_params.config:
        yaml_conf = OmegaConf.load(cli_params.config)
    else:
        yaml_conf = OmegaConf.create()
    
    # Parse the explicit user specified parameters from cli_params
    user_params = {}
    for key in vars(cli_params):        
        cli_value = getattr(cli_params, key)
        if cli_value != getattr(default_params, key):
            user_params[key] = cli_value

    def unflatten_dict(flat_dict):
        """Convert flat dictionary with dot notation keys to nested dictionary."""
        nested = {}
        for key, value in flat_dict.items():
            parts = key.split('.')
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return nested
    
    user_params = unflatten_dict(user_params)
    user_conf = OmegaConf.structured(user_params)

    # Finally resolve the final config
    conf = OmegaConf.create(vars(default_params))
    conf = OmegaConf.merge(conf, yaml_conf)
    conf = OmegaConf.merge(conf, user_conf)
    
    return conf


def main(params):

    # Set up TensorBoard writer, this custom one works only in main process
    tensorboard = CustomTensorboardWriter(
        log_dir=os.path.join(params.output_dir, "tensorboard"))

    # Distributed mode
    udist.init_distributed_mode(params)

    # Set seeds for reproductibility
    seed = params.seed + udist.get_rank()
    # seed = params.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if params.distributed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Print the arguments and add to tensorboard
    print("__git__:{}".format(utils.get_sha()))
    json_params = json.dumps(
        OmegaConf.to_container(params, resolve=True))
    print("__log__:{}".format(json_params))

    # Copy the config files to the output dir
    if udist.is_main_process():
        os.makedirs(os.path.join(params.output_dir, 'configs'), exist_ok=True)
        os.system(f'cp {params.embedder_config} {params.output_dir}/configs/embedder.yaml')
        os.system(f'cp {params.augmentation_config} {params.output_dir}/configs/augs.yaml')
        os.system(f'cp {params.extractor_config} {params.output_dir}/configs/extractor.yaml')

    # Build the embedder model
    embedder_cfg = OmegaConf.load(params.embedder_config)
    params.embedder_model = params.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[params.embedder_model]
    embedder = build_embedder(params.embedder_model, embedder_params, params.nbits, params.hidden_size_multiplier)
    print(embedder)
    print(f'embedder: {sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # build the augmenter
    augmenter_cfg = OmegaConf.load(params.augmentation_config)
    augmenter_cfg.num_augs = params.num_augs
    augmenter = Augmenter(**augmenter_cfg).to(device)
    print(f'augmenter: {augmenter}')

    # Build the extractor model
    extractor_cfg = OmegaConf.load(params.extractor_config)
    params.extractor_model = params.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[params.extractor_model]
    extractor = build_extractor(params.extractor_model, extractor_params, params.img_size_proc, params.nbits)
    print(f'extractor: {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # build attenuation
    if params.attenuation and params.attenuation.lower() != "none":
        attenuation_cfg = OmegaConf.load(params.attenuation_config)
        if params.attenuation.lower().startswith("jnd"):
            attenuation_cfg = OmegaConf.load(params.attenuation_config)
            attenuation = JND(**attenuation_cfg[params.attenuation]).to(device)
        elif params.attenuation.lower().startswith("simplified"):
            attenuation_cfg = OmegaConf.load(params.attenuation_config)
            attenuation = VarianceBasedJND(**attenuation_cfg[params.attenuation]).to(device)
        else:
            attenuation = None
    else:
        attenuation = None
    print(f'attenuation: {attenuation}')

    # Build the autoencoder model.
    if params.autoencoder.lower() != "none":
        autoencoder = getattr(neuralcompression, params.autoencoder)()
        if not params.ae_quantize:
            autoencoder.quantize = lambda x: x  # no quantization
        autoencoder = autoencoder.to(device)
    else:
        autoencoder = None

    # build the complete model
    wam = VideoWam(embedder, extractor, augmenter, attenuation,
                   params.scaling_w, params.scaling_i,
                   img_size=params.img_size_proc,
                   chunk_size=params.videowam_chunk_size,
                   step_size=params.videowam_step_size,
                   blending_method=params.blending_method,
                   lowres_attenuation=params.lowres_attenuation,
                   autoencoder=autoencoder,
                   latent_layer=params.latent_layer,
                   latent_layer_watermarker_input=params.latent_layer_watermarker_input,
                   latent_layer_normalization=params.latent_layer_normalization,
                   ).to(device)
    wam = wam.to(device)
    # print(wam)

    # build losses
    image_detection_loss = DetectionLoss(
        balanced=params.balanced, total_norm=params.total_gnorm,
        disc_weight=params.lambda_d, percep_weight=params.lambda_i,
        detect_weight=params.lambda_det, decode_weight=params.lambda_dec,
        disc_start=params.disc_start, disc_num_layers=params.disc_num_layers, disc_in_channels=params.disc_in_channels,
        percep_loss=params.perceptual_loss, disc_scales=params.disc_scales, maskbit_disc=params.maskbit_disc,
        lecam_regularization_weight=params.lecam_regularization_weight, maskbit_maxpool=params.maskbit_maxpool,
    ).to(device)
    print(image_detection_loss)
    # print(f"discriminator: {sum(p.numel() for p in image_detection_loss.discriminator.parameters() if p.requires_grad) / 1e3:.1f}K parameters")

    # Build the scaling schedule. Default is none
    if params.scaling_w_schedule is not None:
        scaling_w_schedule = uoptim.parse_params(params.scaling_w_schedule)
        scaling_scheduler = uoptim.ScalingScheduler(
            obj=wam.blender, attribute="scaling_w", scaling_o=params.scaling_w,
            **scaling_w_schedule
        )
    else:
        scaling_scheduler = None

    # Build optimizer and scheduler
    model_params = list(embedder.parameters()) + list(extractor.parameters())
    optim_params = uoptim.parse_params(params.optimizer)
    optimizer = uoptim.build_optimizer(model_params, **optim_params)
    scheduler_params = uoptim.parse_params(params.scheduler)
    scheduler = uoptim.build_lr_scheduler(optimizer, **scheduler_params)
    print('optimizer: %s' % optimizer)
    print('scheduler: %s' % scheduler)

    # discriminator optimizer
    if params.optimizer_d is None:
        optim_params_d = uoptim.parse_params(params.optimizer) 
    else:
        optim_params_d = uoptim.parse_params(params.optimizer_d)
    discriminator_params = list(image_detection_loss.discriminator.parameters())
    optimizer_d = uoptim.build_optimizer(
        model_params=discriminator_params,
        **optim_params_d
    )
    scheduler_d = uoptim.build_lr_scheduler(optimizer=optimizer_d, **scheduler_params)
    print('optimizer_d: %s' % optimizer_d)
    print('scheduler_d: %s' % scheduler_d)

    # Data loaders
    train_transform, train_mask_transform = get_resize_transform(params.img_size, resize_only=params.resize_only)
    val_transform, val_mask_transform = get_resize_transform(params.img_size_val)
    image_train_loader = image_val_loader = None

    image_train_loader = get_dataloader_segmentation(params.dataset.train_dir,
                                                        transform=train_transform,
                                                        mask_transform=train_mask_transform,
                                                        batch_size=params.batch_size,
                                                        num_workers=params.workers, shuffle=True)
    image_val_loader = get_dataloader_segmentation(params.dataset.val_dir,
                                                    transform=val_transform,
                                                    mask_transform=val_mask_transform,
                                                    batch_size=params.batch_size_eval,
                                                    num_workers=params.workers,
                                                    shuffle=False)


    # optionally resume training
    if params.resume_from is not None:
        components_to_load = {'model': wam}
        if params.resume_disc:
            components_to_load['discriminator'] = image_detection_loss.discriminator
        if params.resume_optimizer_state:
            components_to_load['optimizer'] = optimizer
            components_to_load['optimizer_d'] = optimizer_d
        uoptim.restart_from_checkpoint(
            params.resume_from,
            **components_to_load
        )

    to_restore = {
        "epoch": 0,
    }
    uoptim.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=wam,
        discriminator=image_detection_loss.discriminator,
        optimizer=optimizer,
        optimizer_d=optimizer_d,
        scheduler=scheduler,
        scheduler_d=scheduler_d,
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']
    for param_group in optimizer_d.param_groups:
        param_group['lr'] = optim_params_d['lr']
    optimizers = [optimizer, optimizer_d]

    # specific thing to do if distributed training
    if params.distributed:
        # if model has batch norm convert it to sync batchnorm in distributed mode
        wam = nn.SyncBatchNorm.convert_sync_batchnorm(wam)
        compilation = "none"
        if compilation == "reduce-overhead":
            wam.autoencoder.decode = torch.compile(wam.autoencoder.decode, mode="reduce-overhead")
            wam.autoencoder.encode_pre_quant = torch.compile(wam.autoencoder.encode_pre_quant, mode="reduce-overhead")
            wam.embedder = torch.compile(wam.embedder, mode="reduce-overhead")
            wam.detector = torch.compile(wam.detector, mode="reduce-overhead")
        elif compilation == "normal":
            wam.autoencoder.decode = torch.compile(wam.autoencoder.decode)
            wam.autoencoder.encode_pre_quant = torch.compile(wam.autoencoder.encode_pre_quant)
            wam.embedder = torch.compile(wam.embedder)
            wam.detector = torch.compile(wam.detector)
        wam_ddp = nn.parallel.DistributedDataParallel(
            wam, device_ids=[params.local_rank], find_unused_parameters=True)
        image_detection_loss.discriminator = nn.parallel.DistributedDataParallel(
            image_detection_loss.discriminator, device_ids=[params.local_rank])
        wam = wam_ddp.module
    else:
        wam_ddp = wam

    dummy_img = torch.ones(3, params.img_size_val, params.img_size_val)
    validation_masks = augmenter.mask_embedder.sample_representative_masks(
        dummy_img)  # n 1 h w, full of ones or random masks depending on config

    # evaluation only
    if params.only_eval and udist.is_main_process():

        augs = get_validation_augs()

        print("running eval on .")
        val_stats = eval_one_epoch(wam, image_val_loader,
                                    0, augs, validation_masks, params)
        with open(os.path.join(params.output_dir, 'log_only_eval.txt'), 'a') as f:
            f.write(json.dumps(val_stats) + "\n")

    # start training
    print('training...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs):

        # scheduler
        if scheduler is not None:
            scheduler.step(epoch)
            scheduler_d.step(epoch)
        if scaling_scheduler is not None:
            scaling_scheduler.step(epoch)

        if params.distributed:
            image_train_loader.sampler.set_epoch(epoch)

        # prepare if freezing the generator and finetuning the detector
        if epoch >= params.finetune_detector_start:
            # remove the grads from embedder
            wam.embedder.requires_grad_(False)
            wam.embedder.eval()

            # Only rebuild DDP if not already wrapped
            # NOTE: Do not recreate DDP if already initialized - it causes process group conflicts
            if params.distributed and not isinstance(wam_ddp, nn.parallel.DistributedDataParallel):
                wam_ddp = nn.parallel.DistributedDataParallel(
                    wam, device_ids=[params.local_rank], find_unused_parameters=True)

            # set to 0 the weights of the perceptual losses
            params.lambda_i = 0.0
            params.lambda_d = 0.0
            params.balanced = False
            image_detection_loss.percep_weight = 0.0
            image_detection_loss.disc_weight = 0.0
            image_detection_loss.balanced = False  # not supported here because embedder is frozen

        # train and log
        train_stats = train_one_epoch(wam_ddp, optimizers, image_train_loader, image_detection_loss, epoch, params, tensorboard=tensorboard)
        log_stats = {
            'epoch': epoch, 
            **{f'train_{k}': v for k, v in train_stats.items()}
        }

        if epoch % params.eval_freq == 0:
            if (epoch % params.full_eval_freq == 0 and epoch > 0) or (epoch == params.epochs-1):
                augs = get_validation_augs()
            else:
                augs = get_validation_augs_subset()
            val_stats = eval_one_epoch(wam, image_train_loader,
                                    epoch, augs, validation_masks, params, tensorboard=tensorboard)
            log_stats = {
                **log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}



        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")
        if udist.is_dist_avail_and_initialized():
            dist.barrier()  # Ensures all processes wait until the main node finishes validation

        print("Saving Checkpoint..")
        discrim_no_ddp = image_detection_loss.discriminator.module if params.distributed else image_detection_loss.discriminator
        save_dict = {
            'epoch': epoch + 1,
            'model': wam.state_dict(),
            'discriminator': discrim_no_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'scheduler_d': scheduler_d.state_dict() if scheduler_d is not None else None,
            'args': OmegaConf.to_yaml(params),
        }
        udist.save_on_master(save_dict, os.path.join(
            params.output_dir, 'checkpoint.pth'))
        if params.saveckpt_freq and epoch % params.saveckpt_freq == 0:
            udist.save_on_master(save_dict, os.path.join(
                params.output_dir, f'checkpoint{epoch:03}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))


def train_one_epoch(
    wam: Wam,
    optimizers: List[torch.optim.Optimizer],
    train_loader: torch.utils.data.DataLoader,
    image_detection_loss: DetectionLoss,
    epoch: int,
    params: argparse.Namespace,
    tensorboard: CustomTensorboardWriter
) -> dict:

    wam.train()

    header = f'Train - Epoch: [{epoch}/{params.epochs}]'
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    for it, batch_items in enumerate(metric_logger.log_every(train_loader, 10, header)):
        if it >= params.iter_per_epoch:
            break

        # some data loaders return batch_data, masks, frames_positions as well
        batch_imgs, batch_masks = batch_items[0], batch_items[1]
        
        assert batch_imgs.ndim == 4, "Only image watermarking is supported in this script"

        accumulation_steps = 1
        batch_masks = batch_masks.unsqueeze(0)
        batch_imgs = batch_imgs.unsqueeze(0)

        if params.lambda_d == 0:  # no disc, optimize embedder/extractor only
            optimizer_ids_for_epoch = [0]
        else:
            if params.sleepwake:  # alternate
                optimizer_ids_for_epoch = [epoch % 2]
            else:
                if epoch < params.disc_start:  # no disc at the beginning
                    optimizer_ids_for_epoch = [0]
                else:
                    optimizer_ids_for_epoch = [1, 0]

        # reset the optimizer gradients before accum gradients
        for optimizer_idx in optimizer_ids_for_epoch:
            optimizers[optimizer_idx].zero_grad()

        # accumulate gradients
        for acc_it in range(accumulation_steps):

            imgs, masks = batch_imgs[acc_it], batch_masks[acc_it]
            imgs = imgs.to(device, non_blocking=True)

            # forward
            outputs = wam(imgs, masks, is_video=False, is_detection_loss=(params.lambda_det > 0))
            outputs["preds"] /= params.temperature

            # last layer is used for gradient scaling
            last_layer = wam.embedder.get_last_layer() if not params.distributed else wam.module.embedder.get_last_layer()

            # index 1 for discriminator, 0 for embedder/extractor
            for optimizer_idx in optimizer_ids_for_epoch:
                if "imgs_ori_autoencoded" in outputs and params.discriminator_on_autoencoded:
                    detection_loss_inputs = outputs["imgs_ori_autoencoded"].detach()
                else:
                    detection_loss_inputs = imgs
                loss, logs = image_detection_loss(
                    detection_loss_inputs, outputs["imgs_w"],
                    outputs["masks"], outputs["msgs"], outputs["preds"],
                    optimizer_idx, epoch,
                    last_layer=last_layer,
                    preds_ori=outputs.get("preds_ori", None),
                )
                # Scale loss for accumulation so lr is not affected
                loss = loss / accumulation_steps
                loss.backward()

            # log stats
            log_stats = {
                **logs,
                'psnr': psnr(outputs["imgs_w"], detection_loss_inputs).mean().item(),
                'ssim': ssim(outputs["imgs_w"], detection_loss_inputs).mean().item(),
                'lr': optimizers[0].param_groups[0]['lr'],
            }

            bit_preds = outputs["preds"][:, 1:]  # b k h w
            mask_preds = outputs["preds"][:, 0:1]  # b 1 h w

            # bit accuracy
            if params.nbits > 0:
                bit_accuracy_ = bit_accuracy(
                    bit_preds,  # b k h w
                    outputs["msgs"],  # b k
                    outputs["masks"]
                ).nanmean().item()
                log_stats['bit_acc'] = bit_accuracy_

            # localization metrics
            if params.lambda_det > 0:
                # iou0 = iou(mask_preds, outputs["masks"], label=0).mean().item()
                # iou1 = iou(mask_preds, outputs["masks"], label=1).mean().item()
                # log_stats.update({
                #     f'acc': accuracy(mask_preds, outputs["masks"]).mean().item(),
                #     f'miou': (iou0 + iou1) / 2,
                # })
                detection_inputs = torch.cat([
                    outputs["preds"][:, 0:1],
                    outputs["preds_ori"][:, 0:1]
                ], dim=0)
                detection_targets = torch.cat([
                    torch.ones_like(outputs["preds"][:, 0:1]),
                    torch.zeros_like(outputs["preds_ori"][:, 0:1])],
                    dim=0)
                acc_ = bit_accuracy(
                    detection_inputs,  # b 1
                    detection_targets,  # b 1
                    outputs["masks"]
                ).nanmean().item()
                log_stats['acc'] = acc_

            torch.cuda.synchronize()
            for name, value in log_stats.items():
                metric_logger.update(**{name: value})

            # save images on training
            if (epoch % params.saveimg_freq == 0) and it == acc_it == 0:
            # if (epoch % params.saveimg_freq == 0) and (it % 50) == 0:
                ori_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_0_ori.png')
                wm_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_1_wm.png')
                diff_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_2_diff.png')
                aug_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_3_aug_{outputs["selected_aug"]}.png')
                autoencoded_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_4_autoencoded.png')
                if udist.is_main_process():
                    save_image(imgs, ori_path, nrow=8)
                    tensorboard.add_images("TRAIN/IMAGES/orig", imgs, epoch)
                    save_image(outputs["imgs_w"], wm_path, nrow=8)
                    tensorboard.add_images(
                        "TRAIN/IMAGES/wmed", outputs["imgs_w"], epoch)
                    if "imgs_ori_autoencoded" in outputs:
                        save_image(create_diff_img(
                            outputs["imgs_ori_autoencoded"], outputs["imgs_w"]), diff_path, nrow=8)
                    else:
                        save_image(create_diff_img(
                            imgs, outputs["imgs_w"]), diff_path, nrow=8)
                    tensorboard.add_images("TRAIN/IMAGES/diff", create_diff_img(
                        imgs, outputs["imgs_w"]), epoch)
                    save_image(outputs["imgs_aug"], aug_path, nrow=8)
                    tensorboard.add_images(
                        "TRAIN/IMAGES/aug", outputs["imgs_aug"], epoch)
                    if "imgs_ori_autoencoded" in outputs:
                        save_image(outputs["imgs_ori_autoencoded"], autoencoded_path, nrow=8)


        # end accumulate gradients batches
        # add optimizer step
        for optimizer_idx in optimizer_ids_for_epoch:
            optimizers[optimizer_idx].step()

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    train_logs = {k: meter.global_avg for k,
                  meter in metric_logger.meters.items()}

    tensorboard.add_scalars("TRAIN/LOSS", train_logs, epoch)

    return train_logs

@ torch.no_grad()
def eval_one_epoch(
    wam: Wam,
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
    validation_augs: List,
    validation_masks: torch.Tensor,
    params: argparse.Namespace,
    tensorboard: Optional[CustomTensorboardWriter] = None,
) -> dict:
    """
    Evaluate the model on the validation set, with different augmentations

    Args:
        wam (Wam): the model
        val_loader (torch.utils.data.DataLoader): the validation loader
        image_detection_loss (DetectionLoss): the loss function
        epoch (int): the current epoch
        validation_augs (List): list of augmentations to apply
        validation_masks (torch.Tensor): the validation masks, full of ones for now
        params (argparse.Namespace): the parameters
    """
    if torch.is_tensor(validation_masks):
        validation_masks = list(torch.unbind(validation_masks, dim=0))

    wam.eval()

    header = f'Val - Epoch: [{epoch}/{params.epochs}]'
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    for it, batch_items in enumerate(metric_logger.log_every(val_loader, 10, header)):
        if params.iter_per_valid is not None and it >= params.iter_per_valid:
            break

        # some data loaders return batch_data, masks, frames_positions as well
        batch_imgs, batch_masks = batch_items[0], batch_items[1]

        assert batch_imgs.ndim == 4, "Only image watermarking are supported in this script"

        accumulation_steps = 1
        batch_masks = batch_masks.unsqueeze(0)  # 1 b 1 h w
        batch_imgs = batch_imgs.unsqueeze(0)  # 1 b c h w

        for acc_it in range(accumulation_steps):
            imgs, masks = batch_imgs[acc_it], batch_masks[acc_it]

            # forward embedder
            embed_time = time.time()
            outputs = wam.embed(imgs, is_video=False, lowres_attenuation=params.lowres_attenuation)
            embed_time = (time.time() - embed_time) / imgs.shape[0]
            msgs = outputs["msgs"].to(device)  # b k
            imgs_w = outputs["imgs_w"]  # b c h w

            if (epoch % params.saveimg_freq == 0) and it == acc_it == 0 and udist.is_main_process():
                base_name = os.path.join(
                    params.output_dir, f'{epoch:03}_{acc_it*it:03}_val')
                ori_path = base_name + '_0_ori.png'
                wm_path = base_name + '_1_wm.png'
                diff_path = base_name + '_2_diff.png'
                save_image(imgs, ori_path, nrow=8)
                save_image(imgs_w, wm_path, nrow=8)
                if "imgs_ori_autoencoded" in outputs:
                    save_image(outputs["imgs_ori_autoencoded"],
                               base_name + '_3_ori_auto.png', nrow=8)
                    save_image(create_diff_img(outputs["imgs_ori_autoencoded"], imgs_w), 
                               diff_path, nrow=8)
                else:
                    save_image(create_diff_img(imgs, imgs_w), diff_path, nrow=8)
                
                if tensorboard is not None:
                    tensorboard.add_images(
                        "VALID/IMAGES/orig", imgs, acc_it*it*epoch)
                    tensorboard.add_images(
                        "VALID/IMAGES/wmed", imgs_w, acc_it*it*epoch)
                    tensorboard.add_images(
                        "VALID/IMAGES/diff", create_diff_img(imgs, imgs_w), acc_it*it*epoch)

            # quality metrics
            metrics = {}
            if "imgs_ori_autoencoded" in outputs:
                metrics['psnr'] = psnr(imgs_w, outputs["imgs_ori_autoencoded"]).mean().item()
                metrics['ssim'] = ssim(imgs_w, outputs["imgs_ori_autoencoded"]).mean().item()
            else:
                metrics['psnr'] = psnr(imgs_w, imgs).mean().item()
                metrics['ssim'] = ssim(imgs_w, imgs).mean().item()
            metrics['embed_time'] = embed_time
            torch.cuda.synchronize()
            metric_logger.update(**metrics)

            extract_times = []
            for mask_id, masks in enumerate(validation_masks):
                # watermark masking
                masks = masks.to(imgs.device)  # 1 h w
                if len(masks.shape) < 4:
                    masks = masks.unsqueeze(0).repeat(
                        imgs_w.shape[0], 1, 1, 1)  # b 1 h w
                imgs_masked = imgs_w * masks + imgs * (1 - masks)

                for transform_instance, strengths in validation_augs:

                    for strength in strengths:
                        imgs_aug, masks_aug = transform_instance(
                                imgs_masked, masks, strength)
                        selected_aug = str(transform_instance) + f"_{strength}"
                        selected_aug = selected_aug.replace(", ", "_")

                        # extract watermark
                        extract_time = time.time()
                        outputs = wam.detect(imgs_aug)
                        extract_time = time.time() - extract_time
                        extract_times.append(extract_time / imgs_aug.shape[0])
                        preds = outputs["preds"]
                        bit_preds = preds[:, 1:]  # b k ...

                        aug_log_stats = {}
                        if params.nbits > 0:
                            bit_accuracy_ = bit_accuracy(
                                bit_preds,
                                msgs.to(bit_preds.device),
                                masks_aug
                            ).nanmean().item()

                        if params.nbits > 0:
                            aug_log_stats[f'bit_acc'] = bit_accuracy_

                        if params.lambda_det > 0:
                            # iou0 = iou(mask_preds, masks,
                            #            label=0).mean().item()
                            # iou1 = iou(mask_preds, masks,
                            #            label=1).mean().item()
                            # aug_log_stats.update({
                            #     f'acc': accuracy(mask_preds, masks).mean().item(),
                            #     f'miou': (iou0 + iou1) / 2,
                            # })
                            imgs_ori_aug, _ = transform_instance(
                                imgs, masks, strength)
                            preds_ori = wam.detect(imgs_ori_aug, is_video=False)["preds"]
                            detection_inputs = torch.cat([
                                preds[:, 0:1],
                                preds_ori[:, 0:1]
                            ], dim=0)
                            detection_targets = torch.cat([
                                torch.ones_like(preds[:, 0:1]),
                                torch.zeros_like(preds_ori[:, 0:1])],
                                dim=0)
                            acc_ = bit_accuracy(
                                detection_inputs,  # b 1
                                detection_targets,  # b 1
                            ).nanmean().item()
                            aug_log_stats[f'acc'] = acc_
                            

                        current_key = f"mask={mask_id}_aug={selected_aug}"
                        aug_log_stats = {f"{k}_{current_key}": v for k,
                                         v in aug_log_stats.items()}

                        torch.cuda.synchronize()
                        metric_logger.update(**aug_log_stats)

            metrics['extract_time'] = np.mean(extract_times)
            torch.cuda.synchronize()
            metric_logger.update(**metrics)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('val'), metric_logger)
    valid_logs = {k: meter.global_avg for k,
                  meter in metric_logger.meters.items()}
    if tensorboard is not None:
        tensorboard.add_scalars("VALID", valid_logs, epoch)
    return valid_logs


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parse_args(parser)
    
    print(f"Params: {OmegaConf.to_yaml(params)}")
    # run experiment
    main(params)
