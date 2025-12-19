from typing import Callable, Optional

import os
import diffusers
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch import nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from distseal.augmentation.maskgit_utils import Encoder as MaskgitEncoder
from distseal.augmentation.maskgit_utils import Decoder as MaskgitDecoder
from distseal.augmentation.maskgit_utils import VectorQuantizer as MaskgitQuantizer
from distseal.utils.dist import is_main_process, is_distributed
from deps.efficientvit.models.efficientvit.dc_ae import DCAE, DCAEConfig, dc_ae_f32c32, dc_ae_f64c128, dc_ae_f128c512
from deps.efficientvit.maskbit import ConvVQModel

__all__ = ["create_dc_ae_model_cfg", "DCAE_HF", "AutoencoderKL"]


REGISTERED_DCAE_MODEL: dict[str, tuple[Callable, Optional[str]]] = {
    "dc-ae-f32c32-in-1.0": (dc_ae_f32c32, None),
    "dc-ae-f64c128-in-1.0": (dc_ae_f64c128, None),
    "dc-ae-f128c512-in-1.0": (dc_ae_f128c512, None),
    #################################################################################################
    "dc-ae-f32c32-mix-1.0": (dc_ae_f32c32, None),
    "dc-ae-f64c128-mix-1.0": (dc_ae_f64c128, None),
    "dc-ae-f128c512-mix-1.0": (dc_ae_f128c512, None),
    #################################################################################################
    "dc-ae-f32c32-sana-1.0": (dc_ae_f32c32, None),
}


def create_dc_ae_model_cfg(name: str, pretrained_path: Optional[str] = None) -> DCAEConfig:
    assert name in REGISTERED_DCAE_MODEL, f"{name} is not supported"
    dc_ae_cls, default_pt_path = REGISTERED_DCAE_MODEL[name]
    pretrained_path = default_pt_path if pretrained_path is None else pretrained_path
    model_cfg = dc_ae_cls(name, pretrained_path)
    return model_cfg


class DCAE_HF(DCAE, PyTorchModelHubMixin):
    def __init__(self, model_name: str):
        cfg = create_dc_ae_model_cfg(model_name)
        DCAE.__init__(self, cfg)


class AutoencoderKL(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        if self.model_name in ["stabilityai/sd-vae-ft-ema"]:
            self.model = diffusers.models.AutoencoderKL.from_pretrained(self.model_name)
            self.spatial_compression_ratio = 8
        elif self.model_name == "flux-vae":
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
            self.model = diffusers.models.AutoencoderKL.from_pretrained(pipe.vae.config._name_or_path)
            self.spatial_compression_ratio = 8
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in ["stabilityai/sd-vae-ft-ema", "flux-vae"]:
            return self.model.encode(x).latent_dist.sample()
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.model_name in ["stabilityai/sd-vae-ft-ema", "flux-vae"]:
            return self.model.decode(latent).sample
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")


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
    
    def forward(self, x: torch.Tensor, global_step: int, watermarker=None, msg=None) -> torch.Tensor:
        x = (x * 0.5) + 0.5  # to [0,1]
        x = self.encoder(x)
        quantize_before = watermarker is not None and watermarker.latent_layer == "input_after_quantize"
        if quantize_before:
            x = self.quantize(x)
        if watermarker is not None and msg is not None and watermarker.latent_watermarker:
            msg_batch = msg.repeat(x.shape[0], 1).to(x.device)
            preds_w = watermarker.embedder(x, msg_batch)
            x = watermarker.blender(x, preds_w)
        if not quantize_before:
            x = self.quantize(x)
        x = self.decoder(x)
        if watermarker is not None and msg is not None and not watermarker.latent_watermarker:
            msg_batch = msg.repeat(x.shape[0], 1).to(x.device)
            x = x.clamp(0, 1)  # to [0,1]
            # if watermarker.embedder.yuv:  # take y channel only
            #     preds_w = watermarker.embedder(watermarker.rgb2yuv(x)[:, 0:1], msg_batch)
            # else:
            #     preds_w = watermarker.embedder(x, msg_batch)
            # x = watermarker.blender(x, preds_w)
            
            x = watermarker.embed(x, msg_batch, is_video=False)["imgs_w"]
        x = x * 2 - 1  # to [-1,1]
        return x, torch.tensor(0), {}


class MaskBitCompression(nn.Module):
    """Base class for MaskBit models"""
    
    def __init__(self, config_dict, checkpoint_path=None, hf_repo_id=None, hf_filename=None):
        super(MaskBitCompression, self).__init__()
        
        # Create a simple config object
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            def get(self, key, default):
                return getattr(self, key, default)
        
        self.config = Config(**config_dict)
        self.checkpoint_path = checkpoint_path
        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename
        
        # Initialize the MaskBit model
        model = ConvVQModel(self.config)
        model.eval()
        
        # Load checkpoint - try HuggingFace first, then local path
        model_loaded = False
        
        if hf_repo_id and hf_filename:
            try:
                # Download model from HuggingFace Hub
                downloaded_path = hf_hub_download(
                    repo_id=hf_repo_id, 
                    filename=hf_filename,
                    local_dir="./",
                    local_dir_use_symlinks=False
                )
                model.load_pretrained(downloaded_path, strict_loading=False)
                print(f"✓ Loaded MaskBit model from HuggingFace: {hf_repo_id}/{hf_filename}")
                model_loaded = True
            except Exception as e:
                print(f"⚠ Failed to load from HuggingFace: {e}")
        
        # Fallback to local checkpoint if provided and HF loading failed
        if not model_loaded and checkpoint_path and os.path.exists(checkpoint_path):
            try:
                model.load_pretrained(checkpoint_path, strict_loading=False)
                print(f"✓ Loaded MaskBit model from local path: {checkpoint_path}")
                model_loaded = True
            except Exception as e:
                print(f"⚠ Failed to load from local path: {e}")
        
        if not model_loaded:
            print("⚠ No pre-trained weights loaded. Using randomly initialized MaskBit model.")
        
        # Set model to eval mode and freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.quantize_fn = model.quantize

    def preprocess(self, x):
        """MaskBit expects input in range [0, 1]"""
        return x

    def postprocess(self, x):
        """MaskBit outputs in range [0, 1]"""
        return torch.clamp(x, 0.0, 1.0)

    def _handle_size_requirements(self, image: torch.Tensor, mask: torch.Tensor = None):
        """Handle input size requirements (ensure dimensions are multiples of 16)"""
        h, w = image.shape[-2:]
        original_size = (h, w)
          
        # Ensure dimensions are multiples of 16
        if h % 16 != 0 or w % 16 != 0:
            h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
            w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
          
        return image, mask, original_size

    def encode_pre_quant(self, image: torch.Tensor):
        """Encode image to pre-quantization latents (stop at norm_out layer)"""
        # Handle input size requirements
        image, _, original_size = self._handle_size_requirements(image)

        # MaskBit expects input in range [0, 1]
        image = self.preprocess(image)

        latent = self.encoder(image)

        return latent, original_size

    def quantize(self, latent: torch.Tensor):
        """Quantize latents using MaskBit quantizer"""

        # Quantize
        quantized, result_dict = self.quantize_fn(latent)
        return quantized

    def decode(self, quant_latent: torch.Tensor, original_size: tuple):
        """Decode quantized latents back to image"""
        # Decode quantized latent
        x_hat = self.decoder(quant_latent)
        
        # Post-process to ensure [0, 1] range
        x_hat = self.postprocess(x_hat)
        
        # Resize back to original if needed
        if original_size != x_hat.shape[-2:]:
            x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)
        
        return x_hat

    def forward(self, x: torch.Tensor, global_step: int, watermarker=None, msg=None):
        """Forward pass through MaskBit model"""
        x = (x * 0.5) + 0.5  # to [0,1]
        x = self.encoder(x)
        quantize_before = watermarker is not None and watermarker.latent_layer == "input_after_quantize"
        if quantize_before:
            x = self.quantize(x)
        if watermarker is not None and msg is not None and watermarker.latent_watermarker:
            msg_batch = msg.repeat(x.shape[0], 1).to(x.device)
            preds_w = watermarker.embedder(x, msg_batch)
            x = watermarker.blender(x, preds_w)
        if not quantize_before:
            x = self.quantize(x)
        x = self.decoder(x)
        if watermarker is not None and msg is not None and not watermarker.latent_watermarker:
            msg_batch = msg.repeat(x.shape[0], 1).to(x.device)
            x = x.clamp(0, 1)  # to [0,1]
            x = watermarker.embed(x, msg_batch, is_video=False)["imgs_w"]
        x = x * 2 - 1  # to [-1,1]
        return x, torch.tensor(0), {}


class MaskBit14Bit(MaskBitCompression):
    """MaskBit tokenizer with 14-bit tokens (lookup-free quantization)"""
    def __init__(self, checkpoint_path=None):
        config_dict = {
            "quantizer_type": "lookup-free",
            "codebook_size": 16384,
            "token_size": 14,
            "commitment_cost": 0.25,
            "entropy_loss_weight": 0.02,
            "entropy_loss_temperature": 0.01,
            "entropy_gamma": 1.0,
            "num_channels": 3,
            "hidden_channels": 128,
            "channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "num_res_blocks": 2,
            "sample_with_conv": True,
        }
        # Default checkpoint path if not provided
        checkpoint_path = "/checkpoint/avseal/models/distseal/pretrained/maskbit_tokenizer_14bit.bin"
        
        # Try HuggingFace Hub first  
        hf_repo_id = "markweber/maskbit_tokenizer_14bit"
        hf_filename = "maskbit_tokenizer_14bit.bin"
        
        super(MaskBit14Bit, self).__init__(config_dict, checkpoint_path, hf_repo_id, hf_filename)
