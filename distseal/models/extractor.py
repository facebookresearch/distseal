# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch import nn
from distseal.modules.convnext import ConvNeXtV2
from distseal.modules.hidden import HiddenDecoder
from distseal.modules.pixel_decoder import PixelDecoder
from distseal.modules.vit import ImageEncoderViT
from distseal.modules.dvmark import DVMarkDecoder

class Extractor(nn.Module):
    """
    Abstract class for watermark detection.
    """

    def __init__(self) -> None:
        super(Extractor, self).__init__()

    def preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        return imgs * 2 - 1
    
    def postprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        return (imgs + 1) / 2

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The predicted masks and/or messages.
        """
        return ...
    
    def detect_watermark(self, imgs: torch.Tensor):
        wm_logits = self.forward(imgs)[:, 1:]
        wm_pred = (wm_logits > 0).int()
        return wm_pred


class SegmentationExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        pixel_decoder: PixelDecoder,
    ) -> None:
        super(SegmentationExtractor, self).__init__()
        self.image_encoder = image_encoder
        self.pixel_decoder = pixel_decoder

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        latents = self.image_encoder(imgs)
        masks = self.pixel_decoder(latents)

        return masks


class DinoExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    def __init__(
        self,
        image_encoder: str,
        hook_indices: list[int],
        pixel_decoder: PixelDecoder,
    ) -> None:
        super(DinoExtractor, self).__init__()
        assert image_encoder in ['dinov2_vits14', 'dinov2_vitb14']
        # vits 384, vitb 768
        self.image_encoder = torch.hub.load(
            'facebookresearch/dinov2', image_encoder)
        self.image_encoder.mask_token = None
        self.hook_indices = hook_indices
        self.pixel_decoder = pixel_decoder

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        latents = self.image_encoder.get_intermediate_layers(
            imgs,
            reshape=True, n=self.hook_indices
        )  # 4 x b c h/f w/f
        latents = torch.cat(latents, dim=1)  # 4 b c h/f w/f -> b 4c h/f w/f
        masks = self.pixel_decoder(latents)

        return masks


class HiddenExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    def __init__(
        self,
        hidden_decoder: HiddenDecoder,
    ) -> None:
        super(HiddenExtractor, self).__init__()
        self.hidden_decoder = hidden_decoder

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        masks = self.hidden_decoder(imgs)
        return masks


class ConvnextExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    def __init__(
        self,
        convnext: ConvNeXtV2,
        pixel_decoder: PixelDecoder,
    ) -> None:
        super(ConvnextExtractor, self).__init__()
        self.convnext = convnext
        self.pixel_decoder = pixel_decoder

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        latents = self.convnext(imgs)  # b c h/f w/f
        masks = self.pixel_decoder(latents)
        return masks


class EfficientViTExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    def __init__(
        self,
        efficientvit_backbone: nn.Module,
        efficientvit_decoder: PixelDecoder,
    ) -> None:
        super(EfficientViTExtractor, self).__init__()
        self.efficientvit_backbone = efficientvit_backbone
        self.efficientvit_decoder = efficientvit_decoder

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        latents = self.efficientvit_backbone(imgs)  # b c h/f w/f
        masks = self.efficientvit_decoder(latents)
        return masks


def build_extractor(name, cfg, img_size, nbits):
    if name.startswith('sam'):
        cfg.encoder.img_size = img_size
        cfg.pixel_decoder.nbits = nbits
        image_encoder = ImageEncoderViT(**cfg.encoder)
        pixel_decoder = PixelDecoder(**cfg.pixel_decoder)
        extractor = SegmentationExtractor(
            image_encoder=image_encoder, pixel_decoder=pixel_decoder)
    elif name.startswith('dino2'):
        image_encoder = cfg.encoder.name
        hook_indices = cfg.encoder.hook_indices
        pixel_decoder = PixelDecoder(**cfg.pixel_decoder)
        extractor = DinoExtractor(image_encoder, hook_indices, pixel_decoder)
    elif name.startswith('hidden'):
        # updates some cfg
        cfg.num_bits = nbits
        # build the encoder, decoder and msg processor
        hidden_decoder = HiddenDecoder(**cfg)
        extractor = HiddenExtractor(hidden_decoder)
    elif name.startswith('convnext'):
        # updates some cfg
        cfg.pixel_decoder.nbits = nbits
        # build the encoder, decoder and msg processor
        convnext = ConvNeXtV2(**cfg.encoder)
        pixel_decoder = PixelDecoder(**cfg.pixel_decoder)
        extractor = ConvnextExtractor(convnext, pixel_decoder)
    elif name.startswith('efficientvit'):
        
        try:
            from distseal.modules.efficientvit_decoder import EfficientViTDecoder
            from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0, efficientvit_backbone_b1, efficientvit_backbone_l0, efficientvit_backbone_l1
        except ImportError:
            print("EfficientViT not found. Make sure to install the efficientvit package.")

        
        # build the encoder, decoder and msg processor
        image_encoder = cfg.encoder.name
        
        if image_encoder in ['b0', 'b1']:
            if image_encoder == 'b0':
                efficientvit_backbone = efficientvit_backbone_b0()
            elif image_encoder == 'b1':
                efficientvit_backbone = efficientvit_backbone_b1()
            if cfg.encoder.higher_res:
                efficientvit_backbone.stages[0].op_list[0].main.depth_conv.conv.stride = (1, 1)
        else:
            if cfg.encoder.with_att:
                kwargs = {"block_list" : ["res", "fmb", "fmb", "mb", "att"]}
            else:
                kwargs = {"block_list" : ["res", "fmb", "fmb", "mb", "mb"]}
            if image_encoder == 'l0':
                efficientvit_backbone = efficientvit_backbone_l0(**kwargs)
            elif image_encoder == 'l1':
                efficientvit_backbone = efficientvit_backbone_l1(**kwargs)
        efficientvit_decoder = EfficientViTDecoder(backbone_type=image_encoder, nbits=nbits)
        extractor = EfficientViTExtractor(efficientvit_backbone, efficientvit_decoder)
    elif name.startswith("dvmark"):
        extractor = DVMarkDecoder(nbits)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return extractor
