# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Run with:
    python -m distseal.augmentation.neuralcompression
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from distseal.modules.maskgit_utils import Encoder as MaskgitEncoder
from distseal.modules.maskgit_utils import Decoder as MaskgitDecoder
from distseal.modules.maskgit_utils import VectorQuantizer as MaskgitQuantizer
from distseal.utils.dist import is_main_process, is_distributed
from huggingface_hub import hf_hub_download

from omegaconf import OmegaConf

try:
    import compressai
    from compressai.zoo import models as compressai_models
    COMPRESSAI_AVAILABLE = True
except ImportError:
    COMPRESSAI_AVAILABLE = False
    # print("CompressAI package not found. Install with pip install compressai")

try:
    from diffusers import VQModel, AutoencoderKL, AutoencoderDC
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Diffusers package not found. Install with pip install diffusers")

try:
    from taming.models.vqgan import VQModel, GumbelVQ
    from omegaconf import OmegaConf
    TAMING_AVAILABLE = True
except ImportError:
    TAMING_AVAILABLE = False
    # print("Taming Transformers not found. Install for VQGAN support")


compression_model_paths = {
    'vqgan-1024': {
        'config': '/checkpoint/pfz/projects/autoencoders/ldm/vqgan_imagenet_f16_1024/configs/model_noloss.yaml',
        'ckpt': '/checkpoint/pfz/projects/autoencoders/ldm/vqgan_imagenet_f16_1024/checkpoints/last.ckpt'
    },
    'vqgan-16384': {
        'config': '/checkpoint/pfz/projects/autoencoders/ldm/vqgan_imagenet_f16_16384/configs/model_noloss.yaml',
        'ckpt': '/checkpoint/pfz/projects/autoencoders/ldm/vqgan_imagenet_f16_16384/checkpoints/last.ckpt'
    }
}


def get_model(model_name, quality):
    if model_name in compressai_models:
        return compressai_models[model_name](quality=quality, pretrained=True)
    else:
        avail_models = list(compressai_models.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {avail_models}")


def get_diffusers_model(model_id):
    """Load a model from the Diffusers library"""
    if 'vqgan' in model_id.lower() or 'vq-' in model_id.lower():
        model = VQModel.from_pretrained(model_id)
    else:
        model = AutoencoderKL.from_pretrained(model_id)
    return model


def load_vqgan_from_config(config_path, ckpt_path, is_gumbel=False):
    """Load a VQGAN model from config and checkpoint paths"""
    config = OmegaConf.load(config_path)
    if is_gumbel:
        # We don't have any GumbelVQ for now.
        model = GumbelVQ(**config.model.params)
    else:
        # Default to VQModel.
        model = VQModel(**config.model.params)
    
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    
    return model.eval()


class NeuralCompression(nn.Module):
    def __init__(self, model_name, quality):
        super(NeuralCompression, self).__init__()
        self.model_name = model_name
        self.quality = quality
        self.model = get_model(model_name, quality)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        if self.model_name not in ['bmshj2018-factorized']:
            # resize to closest multiple of 64
            h, w = image.shape[-2:]
            h = max((h // 64) * 64, 64)
            w = max((w // 64) * 64, 64)
            if image.shape[-2:] != (h, w):
                image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
                if mask is not None:
                    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        x_hat = self.model(image.to('cpu'))['x_hat'].to(image.device)
        return x_hat, mask
    
    def __repr__(self):
        return f"{self.model_name} (q={self.quality})"


class DiffusersCompression(nn.Module):
    """Base class for models from the Diffusers library"""
    def __init__(self, model_id):
        super(DiffusersCompression, self).__init__()
        self.model_id = model_id
        self.model = get_diffusers_model(model_id)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        # Handle input size requirements if any
        h, w = image.shape[-2:]
        original_size = (h, w)
        
        # Some diffusers models require dimensions to be multiples of 16
        if h % 16 != 0 or w % 16 != 0:
            h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
            w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        
        # VQModel and AutoencoderKL have different API
        if isinstance(self.model, VQModel):
            # For VQModel
            encoded = self.model.encode(image)
            if isinstance(encoded, tuple):
                # Some models return a tuple of (z, indices)
                z = encoded[0]
            else:
                z = encoded
            x_hat = self.model.decode(z)
        else:
            # For AutoencoderKL
            x_hat = self.model.decode(self.model.encode(image).latent_dist.sample()).sample
        
        # Resize back to original if needed
        if original_size != (h, w):
            x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=False)
                
        return x_hat, mask
    
    def __repr__(self):
        return f"Diffusers-{self.model_id.split('/')[-1]}"


class TamingVQGANCompression(nn.Module):
    """Base class for VQGAN models from Taming Transformers"""
    def __init__(self, config_path, ckpt_path, is_gumbel=False):
        super(TamingVQGANCompression, self).__init__()
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.is_gumbel = is_gumbel
        self.model = load_vqgan_from_config(config_path, ckpt_path, is_gumbel)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Split checkpoint path by "/" and get part that contains "vqgan".
        self.model_name = next((part for part in self.ckpt_path.split('/') if 'vqgan' in part.lower()), 'vqgan')

    def preprocess(self, x):
        """Preprocess image to VQGAN input format [-1, 1]"""
        return 2.0 * x - 1.0

    def postprocess(self, x):
        """Convert VQGAN output back to [0, 1]"""
        return (x + 1.0) / 2.0

    def encode_pre_quant(self, image: torch.Tensor):
        # Handle input size requirements for VQGAN (multiple of 16)
        h, w = image.shape[-2:]
        original_size = (h, w)
        
        if h % 16 != 0 or w % 16 != 0:
            h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
            w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
        
        # VQGAN expects input in range [-1, 1]
        image = self.preprocess(image)
        
        # Encode and decode with VQGAN
        latent = self.model.encoder(image)
        latent = self.model.quant_conv(latent)
        return latent, original_size

    def quantize(self, latent: torch.Tensor):
        quant, _, _ = self.model.quantize(latent)
        return quant

    def decode(self, quant_latent: torch.Tensor, original_size: tuple):
        # Decode quantized latent.
        x_hat = self.model.decode(quant_latent)
        
        # Convert back to [0, 1] range.
        x_hat = self.postprocess(x_hat)
        
        # Resize back to original if needed
        if original_size != x_hat.shape[-2:]:
            x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)
                
        return x_hat

    def forward(self, image: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        # Handle input size requirements for VQGAN (multiple of 16)
        h, w = image.shape[-2:]
        original_size = (h, w)
        
        if h % 16 != 0 or w % 16 != 0:
            h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
            w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        
        # VQGAN expects input in range [-1, 1]
        image = self.preprocess(image)
        
        # Encode and decode with VQGAN
        z, _, _ = self.model.encode(image)
        x_hat = self.model.decode(z)
        
        # Convert back to [0, 1] range
        x_hat = self.postprocess(x_hat)
        
        # Resize back to original if needed
        if original_size != (h, w):
            x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=False)
                
        return x_hat, mask
    
    def __repr__(self):
        return f"VQGAN-{self.model_name}"


class DCAutoencoder(nn.Module):
    """Base class for DC autoencoders."""
    def __init__(self, model_id, ckpt_path=None):
        super(DCAutoencoder, self).__init__()
        self.model_id = model_id
        self.model = AutoencoderDC.from_pretrained(model_id)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        """Preprocess image to DC autoencoder input format [-1, 1]"""
        return 2.0 * x - 1.0

    def postprocess(self, x):
        """Convert DC autoencoder output back to [0, 1]"""
        return (x + 1.0) / 2.0

    def encode_pre_quant(self, image: torch.Tensor):
        # Handle input size requirements for DC autoencoder (multiple of 16)
        h, w = image.shape[-2:]
        original_size = (h, w)

        if h % 16 != 0 or w % 16 != 0:
            h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
            w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)

        # DC autoencoder expects input in range [-1, 1]
        image = self.preprocess(image)

        # Encode and decode with DC encoder
        latent = self.model.encode(image).latent
        return latent, original_size

    def quantize(self, latent: torch.Tensor):
        return latent  # DC autoencoder does not quantize, return as is

    def decode(self, quant_latent: torch.Tensor, original_size: tuple):
        # Decode quantized latent.
        x_hat = self.model.decode(quant_latent).sample

        # Convert back to [0, 1] range.
        x_hat = self.postprocess(x_hat)

        # Resize back to original if needed
        if original_size != x_hat.shape[-2:]:
            x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)

        return x_hat


class MaskgitVqgan(nn.Module):
    """Base class for MaskgitVqgan autoencoders."""
    def __init__(self):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = MaskgitEncoder(conf)
        self.decoder = MaskgitDecoder(conf)
        self.quantizer = MaskgitQuantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Download weights in the master process.
        pretrained_weight = "maskgit-vqgan-imagenet-f16-256.bin"
        if is_main_process():
            hf_hub_download(repo_id="fun-research/TiTok", filename=pretrained_weight, local_dir="./")
        if is_distributed():
            torch.distributed.barrier()
        
        # Load pretrained weights
        state_dict = torch.load(pretrained_weight, map_location=torch.device("cpu"))
        # Rename keys containing 'quantize' to 'quantizer'
        state_dict = {k.replace('quantize', 'quantizer'): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=True)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x

    def encode_pre_quant(self, image: torch.Tensor):
        # Handle input size requirements (multiple of 16).
        h, w = image.shape[-2:]
        original_size = (h, w)

        if h % 16 != 0 or w % 16 != 0:
            h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
            w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)

        # Autoencoder expects input in range [0, 1].
        latent = self.encoder(image)
        return latent, original_size

    def quantize(self, latent: torch.Tensor):
        quant, _, _ = self.quantizer(latent)
        return quant

    def decode(self, quant_latent: torch.Tensor, original_size: tuple):
        # Decode quantized latent to image with [0, 1] range.
        x_hat = self.decoder(quant_latent)
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        # Resize back to original if needed
        if original_size != x_hat.shape[-2:]:
            x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)

        return x_hat


class StableDiffusionVAE(DiffusersCompression):
    def __init__(self):
        super(StableDiffusionVAE, self).__init__("stabilityai/sd-vae-ft-ema")


class StableDiffusionXLVAE(DiffusersCompression):
    def __init__(self):
        super(StableDiffusionXLVAE, self).__init__("madebyollin/sdxl-vae-fp16-fix")


class BMSHJ2018Hyperprior(NeuralCompression):
    def __init__(self, quality):
        super(BMSHJ2018Hyperprior, self).__init__("bmshj2018-hyperprior", quality)


class BMSHJ2018Factorized(NeuralCompression):
    def __init__(self, quality):
        super(BMSHJ2018Factorized, self).__init__("bmshj2018-factorized", quality)


class MBT2018Mean(NeuralCompression):
    def __init__(self, quality):
        super(MBT2018Mean, self).__init__("mbt2018-mean", quality)


class MBT2018(NeuralCompression):
    def __init__(self, quality):
        super(MBT2018, self).__init__("mbt2018", quality)


class Cheng2020Anchor(NeuralCompression):
    def __init__(self, quality):
        super(Cheng2020Anchor, self).__init__("cheng2020-anchor", quality)


class Cheng2020Attn(NeuralCompression):
    def __init__(self, quality):
        super(Cheng2020Attn, self).__init__("cheng2020-attn", quality)


class VQGAN1024(TamingVQGANCompression):
    """VQGAN model with 1024 codes"""
    def __init__(self):
        config_path = compression_model_paths['vqgan-1024']['config']
        ckpt_path = compression_model_paths['vqgan-1024']['ckpt']
        super(VQGAN1024, self).__init__(config_path, ckpt_path)


class VQGAN16384(TamingVQGANCompression):
    """VQGAN model with 16384 codes"""
    def __init__(self):
        config_path = compression_model_paths['vqgan-16384']['config']
        ckpt_path = compression_model_paths['vqgan-16384']['ckpt']
        super(VQGAN16384, self).__init__(config_path, ckpt_path)


class DCAEf64c128(DCAutoencoder):
    """DC autoencoder model with f=64 and c=128"""
    def __init__(self):
        model_id = "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers"
        super(DCAEf64c128, self).__init__(model_id)


class DCAEf32c32(DCAutoencoder):
    """DC autoencoder model with f=32 and c=32"""
    def __init__(self):
        model_id = "mit-han-lab/dc-ae-f32c32-in-1.0-diffusers"
        super(DCAEf32c32, self).__init__(model_id)
