# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Test with:
    python -m distseal.models.wam
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..augmentation.augmenter import Augmenter
from ..data.transforms import RGB2YUV
from ..modules.jnd import JND
from .blender import Blender
from .embedder import Embedder
from .extractor import Extractor


class Wam(nn.Module):

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.parameters()).device

    def __init__(
        self,
        embedder: Embedder,
        detector: Extractor,
        augmenter: Augmenter,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        clamp: bool = True,
        img_size: int = 256,
        blending_method: str = "additive",
        autoencoder: nn.Module = None,
        latent_layer: str = "input",
        latent_layer_watermarker_input: bool = True,
        latent_layer_normalization: bool = False,
    ) -> None:
        """
        WAM (watermark-anything models) model that combines an embedder, a detector, and an augmenter.
        Embeds a message into an image and detects it as a mask.

        Arguments:
            embedder: The watermark embedder
            detector: The watermark detector
            augmenter: The image augmenter
            attenuation: The JND model to attenuate the watermark distortion
            scaling_w: The scaling factor for the watermark
            scaling_i: The scaling factor for the image
            img_size: The size at which the images are processed
            clamp: Whether to clamp the output images to [0, 1]
            autoencoder: The autoencoder model to watermark in the latent space
            latent_layer: Which latent layer to use for watermarking in the autoencoder.
        """
        super().__init__()
        # modules
        self.embedder = embedder
        self.detector = detector
        self.augmenter = augmenter
        # image format
        self.img_size = img_size
        self.rgb2yuv = RGB2YUV()
        # blending
        assert blending_method in Blender.AVAILABLE_BLENDING_METHODS
        self.blender = Blender(scaling_i, scaling_w, blending_method)
        self.attenuation = attenuation
        self.clamp = clamp
        self.autoencoder = autoencoder
        self.latent_layer = latent_layer
        self.latent_layer_watermarker_input = latent_layer_watermarker_input
        self.latent_layer_normalization = latent_layer_normalization

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.embedder.get_random_msg(bsz, nb_repetitions)  # b x k

    def forward(
        self,
        imgs: torch.Tensor,
        masks: torch.Tensor,
        msgs: torch.Tensor = None,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
        is_detection_loss: bool = False,
    ) -> dict:
        """
        Does the full forward pass of the WAM model (used for training).
        (1) Generates watermarked images from the input images and messages.
        (2) Augments the watermarked images.
        (3) Detects the watermark in the augmented images.
        """
        # optionally create message
        if msgs is None:
            msgs = self.get_random_msg(imgs.shape[0])  # b x k
            msgs = msgs.to(imgs.device)

        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     **interpolation)

        # generate watermarked images
        if self.autoencoder is not None:
            # Converting to the latent space.
            with torch.no_grad():
                latent_ori, original_size = self.autoencoder.encode_pre_quant(imgs_res)
                latent = latent_ori.clone()

            # Watermarking in the latent space.
            if self.latent_layer.startswith("input"):
                if self.latent_layer == "input_after_quantize":
                    with torch.no_grad():
                        latent = self.autoencoder.quantize(latent)
                preds_w = self.embedder(latent, msgs)
                latent_w = self.blender(latent, preds_w)

                # Converting back to the image space.
                if self.latent_layer == "input_after_quantize":
                    quant_latent_w = latent_w
                else:
                    quant_latent_w = self.autoencoder.quantize(latent_w)
                imgs_w = self.autoencoder.decode(quant_latent_w, original_size)
            else:
                # Watermarking an intermediate layer of the latent decoder.
                quant_latent = self.autoencoder.quantize(latent)
                with torch.no_grad():
                    post_quant_latent = self.autoencoder.model.post_quant_conv(quant_latent)
                    intermed_latent = self.autoencoder.model.decoder.first_part(post_quant_latent, self.latent_layer)
                if self.latent_layer_watermarker_input:
                    preds_w = self.embedder(intermed_latent, msgs)
                else:
                    preds_w = self.embedder(quant_latent, msgs)
                if self.latent_layer_normalization:
                    std = intermed_latent.std([0, 2, 3], keepdim=True)
                    intermed_latent_w = self.blender(intermed_latent, preds_w * std)
                else:
                    intermed_latent_w = self.blender(intermed_latent, preds_w)
                imgs_w = self.autoencoder.model.decoder.second_part(intermed_latent_w, self.latent_layer)

                # Convert back to [0, 1] range.
                imgs_w = self.autoencoder.postprocess(imgs_w)
                
                # Resize back to original if needed
                if original_size != imgs_w.shape[-2:]:
                    x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)

            # Interpolate if needed.
            if imgs.shape[-2:] != (self.img_size, self.img_size):
                imgs_w = F.interpolate(imgs_w, size=imgs.shape[-2:],
                                        **interpolation)
            imgs_w = imgs_w.to(imgs.device)
            if self.clamp:
                imgs_w = torch.clamp(imgs_w, 0, 1)
            
            # Autoencoding of the original images.
            with torch.no_grad():
                quant_latent = self.autoencoder.quantize(latent)
                # quant_latent = latent
                imgs_ori_autoencoded = self.autoencoder.decode(quant_latent, original_size)
                # Interpolate if needed.
                if imgs.shape[-2:] != (self.img_size, self.img_size):
                    imgs_ori_autoencoded = F.interpolate(imgs_ori_autoencoded, size=imgs.shape[-2:],
                                            **interpolation)
                imgs_ori_autoencoded = imgs_ori_autoencoded.to(imgs.device)
                imgs_ori_autoencoded = torch.clamp(imgs_ori_autoencoded, 0, 1)
        else:
            if self.embedder.yuv:  # take y channel only
                preds_w = self.embedder(self.rgb2yuv(imgs_res)[:, 0:1], msgs)
            else:
                preds_w = self.embedder(imgs_res, msgs)

            # interpolate back
            if imgs.shape[-2:] != (self.img_size, self.img_size):
                preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                        **interpolation)
            preds_w = preds_w.to(imgs.device)
            imgs_w = self.blender(imgs, preds_w)

            # apply attenuation and clamp
            if self.attenuation is not None:
                self.attenuation.to(imgs.device)
                imgs_w = self.attenuation(imgs, imgs_w)
            if self.clamp:
                imgs_w = torch.clamp(imgs_w, 0, 1)

        # augment
        original_masks = masks.clone()
        imgs_aug, masks, selected_aug = self.augmenter(
            imgs_w, imgs, original_masks, is_video=False, do_resize=False)

        # interpolate back
        if imgs_aug.shape[-2:] != (self.img_size, self.img_size):
            imgs_aug = F.interpolate(imgs_aug, size=(self.img_size, self.img_size),
                                        **interpolation)

        # detect watermark
        preds = self.detector(imgs_aug)

        # Run the detector on the augmented original images.
        if is_detection_loss:
            if self.autoencoder is not None:
                imgs_ori = torch.cat([imgs, imgs_ori_autoencoded], dim=0)
                original_masks = torch.cat([original_masks, original_masks], dim=0)
            else: 
                imgs_ori = imgs

            imgs_ori_aug, _, _ = self.augmenter(
                imgs_ori, imgs_ori, original_masks, is_video=False, do_resize=False)
            if imgs_ori_aug.shape[-2:] != (self.img_size, self.img_size):
                imgs_ori_aug = F.interpolate(imgs_ori_aug, size=(self.img_size, self.img_size),
                                            **interpolation)
            preds_ori = self.detector(imgs_ori_aug)

        # create and return outputs
        outputs = {
            "msgs": msgs,  # original messages: b k
            "masks": masks,  # augmented masks: b 1 h w
            "preds_w": preds_w,  # predicted watermarks: b c h w
            "imgs_w": imgs_w,  # watermarked images: b c h w
            "imgs_aug": imgs_aug,  # augmented images: b c h w
            "preds": preds,  # predicted masks and/or messages: b (1+nbits) h w
            "selected_aug": selected_aug,  # selected augmentation
        }
        if self.autoencoder is not None:
            outputs["imgs_ori_autoencoded"] = imgs_ori_autoencoded  # autoencoded original images: b c h w
        if is_detection_loss:
            outputs["preds_ori"] = preds_ori  # predicted masks and/or messages: b (1+nbits) h w
        return outputs

    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
        lowres_attenuation: bool = False,
    ) -> dict:
        """
        Generates watermarked images from the input images and messages (used for inference).
        Images may be arbitrarily sized.
        Args:
            imgs (torch.Tensor): Batched images with shape BxCxHxW.
            msgs (torch.Tensor): Optional messages with shape BxK.
            interpolation (dict): Interpolation parameters.
            lowres_attenuation (bool): Whether to attenuate the watermark at low resolution,
                which is more memory efficient for high-resolution images.
        Returns:
            dict: A dictionary with the following keys:
                - msgs (torch.Tensor): Original messages with shape BxK.
                - preds_w (torch.Tensor): Predicted watermarks with shape BxCxHxW.
                - imgs_w (torch.Tensor): Watermarked images with shape BxCxHxW.
        """
        # optionally create message
        if msgs is None:
            msgs = self.get_random_msg(imgs.shape[0])  # b x k

        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     **interpolation)
        imgs_res = imgs_res.to(self.device)

        # generate watermarked images
        if self.autoencoder is not None:
            # Converting to the latent space.
            latent_ori, original_size = self.autoencoder.encode_pre_quant(imgs_res)
            latent = latent_ori.clone()

            # Watermarking in the latent space.
            if self.latent_layer.startswith("input"):
                if self.latent_layer == "input_after_quantize":
                    with torch.no_grad():
                        latent = self.autoencoder.quantize(latent)
                preds_w = self.embedder(latent, msgs.to(self.device))
                latent_w = self.blender(latent, preds_w)

                # Converting back to the image space.
                if self.latent_layer == "input_after_quantize":
                    quant_latent_w = latent_w
                else:
                    quant_latent_w = self.autoencoder.quantize(latent_w)
                # quant_latent_w = latent_w
                imgs_w = self.autoencoder.decode(quant_latent_w, original_size)
            else:
                # Watermarking an intermediate layer of the latent decoder.
                quant_latent = self.autoencoder.quantize(latent)
                with torch.no_grad():
                    post_quant_latent = self.autoencoder.model.post_quant_conv(quant_latent)
                    intermed_latent = self.autoencoder.model.decoder.first_part(post_quant_latent, self.latent_layer)
                if self.latent_layer_watermarker_input:
                    preds_w = self.embedder(intermed_latent, msgs.to(self.device))
                else:
                    preds_w = self.embedder(quant_latent, msgs.to(self.device))
                if self.latent_layer_normalization:
                    std = intermed_latent.std([0, 2, 3], keepdim=True)
                    intermed_latent_w = self.blender(intermed_latent, preds_w * std)
                else:
                    intermed_latent_w = self.blender(intermed_latent, preds_w)
                imgs_w = self.autoencoder.model.decoder.second_part(intermed_latent_w, self.latent_layer)

                # Convert back to [0, 1] range.
                imgs_w = self.autoencoder.postprocess(imgs_w)
                
                # Resize back to original if needed
                if original_size != imgs_w.shape[-2:]:
                    x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)

            # Interpolate if needed.
            if imgs.shape[-2:] != (self.img_size, self.img_size):
                imgs_w = F.interpolate(imgs_w, size=imgs.shape[-2:],
                                        **interpolation)
            imgs_w = imgs_w.to(imgs.device)
            if self.clamp:
                imgs_w = torch.clamp(imgs_w, 0, 1)

            # Autoencoding of the original images.
            with torch.no_grad():
                quant_latent = self.autoencoder.quantize(latent)
                # quant_latent = latent
                imgs_ori_autoencoded = self.autoencoder.decode(quant_latent, original_size)
                # Interpolate if needed.
                if imgs.shape[-2:] != (self.img_size, self.img_size):
                    imgs_ori_autoencoded = F.interpolate(imgs_ori_autoencoded, size=imgs.shape[-2:],
                                            **interpolation)
                imgs_ori_autoencoded = imgs_ori_autoencoded.to(imgs.device)
                imgs_ori_autoencoded = torch.clamp(imgs_ori_autoencoded, 0, 1)
        else:
            if self.embedder.yuv:  # take y channel only
                preds_w = self.embedder(
                    self.rgb2yuv(imgs_res)[:, 0:1],
                    msgs.to(self.device)
                )
            else:
                preds_w = self.embedder(imgs_res, msgs.to(self.device))

            # attenuate at low resolution if needed
            if self.attenuation is not None and lowres_attenuation:
                self.attenuation.to(imgs_res.device)
                hmaps = self.attenuation.heatmaps(imgs_res)
                preds_w = hmaps * preds_w

            # interpolate back
            if imgs.shape[-2:] != (self.img_size, self.img_size):
                preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                        **interpolation)
            preds_w = preds_w.to(imgs.device)
            
            # apply attenuation
            if self.attenuation is not None and not lowres_attenuation:
                self.attenuation.to(imgs.device)
                hmaps = self.attenuation.heatmaps(imgs)
                preds_w = hmaps * preds_w

            # blend and clamp
            imgs_w = self.blender(imgs, preds_w)
            if self.clamp:
                imgs_w = torch.clamp(imgs_w, 0, 1)

        outputs = {
            "msgs": msgs,  # original messages: b k
            "preds_w": preds_w,  # predicted watermarks: b c h w
            "imgs_w": imgs_w,  # watermarked images: b c h w
        }
        if self.autoencoder is not None:
            outputs["imgs_ori_autoencoded"] = imgs_ori_autoencoded  # autoencoded original images: b c h w
        return outputs

    def detect(
        self,
        imgs: torch.Tensor,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    ) -> dict:
        """
        Performs the forward pass of the detector only (used at inference).
        Rescales the input images to 256x256 pixels and then computes the mask and the message.
        Args:
            imgs (torch.Tensor): Batched images with shape BxCxHxW.
        Returns:
            dict: A dictionary with the following keys:
                - preds (torch.Tensor): Predicted masks and/or messages with shape Bx(1+nbits)xHxW.
        """

        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                        **interpolation)
        imgs_res = imgs_res.to(self.device)

        # detect watermark
        preds = self.detector(imgs_res).to(imgs.device)

        outputs = {
            "preds": preds,  # predicted masks and/or messages: b (1+nbits) h w
        }
        return outputs

if __name__ == "__main__":
    import torch
    from ..models.wam import Wam
    from ..models.extractor import Extractor
    from ..augmentation.augmenter import Augmenter
    from ..augmentation.neuralcompression import VQGAN16384
    from ..modules.jnd import JND
    from ..models.blender import Blender
    from ..models.embedder import UnetEmbedder
    from ..modules.msg_processor import MsgProcessor
    from ..modules.unet import UNetMsg
    from ..data.transforms import RGB2YUV


    class DummyExtractor(torch.nn.Module):
        def forward(self, imgs):
            # Output shape: Bx2xHxW (mask + dummy message)
            b, _, h, w = imgs.shape
            return torch.randn(b, 2, h, w, device=imgs.device)

    class DummyAugmenter(torch.nn.Module):
        def forward(self, imgs_w, imgs, masks, is_video=False, do_resize=False):
            # Return augmented images, masks, and a dummy selected_aug
            return imgs_w, masks, "dummy_aug"

    nbits = 64
    hidden_size = 16
    in_channels = 128
    out_channels = 128
    z_channels = 128
    z_channels_mults = (1,)
    msg_processor = MsgProcessor(
        nbits=nbits,
        hidden_size=hidden_size,
        msg_processor_type="binary+concat"
    )
    unetmsg = UNetMsg(
        msg_processor=msg_processor,
        in_channels=in_channels,
        out_channels=out_channels,
        z_channels=z_channels,
        num_blocks=8,
        activation="relu",
        normalization="batch",
        z_channels_mults=z_channels_mults,
        just_upsampling=False,
    )
    embedder = UnetEmbedder(unetmsg, msg_processor)
    detector = DummyExtractor()
    augmenter = DummyAugmenter()
    wam = Wam(
        embedder=embedder,
        detector=detector,
        augmenter=augmenter,
        attenuation=None,
        scaling_w=1.0,
        scaling_i=1.0,
        clamp=True,
        img_size=256,
        blending_method="additive",
        autoencoder=VQGAN16384(),
        latent_layer="upsample_1",
        latent_layer_watermarker_input=True,
        latent_layer_normalization=True
    )
    # bla = embedder(torch.rand(2, 128, 16, 16), wam.get_random_msg(2))
    # print("Output shape from embedder:", bla.shape)
    imgs = torch.rand(2, 3, 256, 256)
    masks = torch.ones(2, 1, 256, 256)
    msgs = wam.get_random_msg(2)
    out = wam(imgs, masks, msgs)
    print("Output keys:", out.keys())
    assert "imgs_w" in out
    assert out["imgs_w"].shape == imgs.shape
    assert out["preds"].shape[0] == imgs.shape[0]

