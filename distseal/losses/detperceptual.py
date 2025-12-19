# Copyright (c) Meta Platforms, Inc. and affiliates.
# adapted from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.discriminator import NLayerDiscriminator, NLayerDiscriminatorv2
from ..utils.optim import freeze_grads
from .perceptual import PerceptualLoss

def hinge_d_loss(logits_real, logits_fake):
    """
    https://paperswithcode.com/method/gan-hinge-loss
    """
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    """
    Adopt weight if global step is less than threshold
    """
    if global_step < threshold:
        weight = value
    return weight

def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DetectionLoss(nn.Module):
    def __init__(self,
                 balanced=True, total_norm=0.0,
                 disc_weight=1.0, percep_weight=1.0, detect_weight=1.0, decode_weight=0.0,
                 disc_start=0, disc_num_layers=3, disc_in_channels=3, disc_loss="hinge", use_actnorm=False,
                 percep_loss="lpips", disc_scales=1,
                 maskbit_disc=False, maskbit_maxpool=16, lecam_regularization_weight=0.0,
                 ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]

        self.balanced = balanced
        self.total_norm = total_norm

        self.percep_weight = percep_weight
        self.detect_weight = detect_weight
        self.disc_weight = disc_weight
        self.decode_weight = decode_weight

        # self.perceptual_loss = PerceptualLoss(percep_loss=percep_loss).to(torch.device("cuda"))
        self.perceptual_loss = PerceptualLoss(percep_loss=percep_loss)

        def make_discriminator():
            """Choose which version of the discriminator to use."""
            if maskbit_disc:
                return NLayerDiscriminatorv2(
                    num_channels=disc_in_channels,
                    num_stages=disc_num_layers,
                    maxpool=maskbit_maxpool
                )
            else:
                return NLayerDiscriminator(
                    input_nc=disc_in_channels,
                    n_layers=disc_num_layers,
                    use_actnorm=use_actnorm
                ).apply(weights_init)
        
        class MultiscaleDisc(nn.Module):
            def __init__(self, disc_scales):
                super().__init__()
                self.discriminators = nn.ModuleDict({str(2**i): make_discriminator() for i in range(disc_scales)})

            def forward(self, inputs):
                logits = []
                for scale, disc in self.discriminators.items():
                    if scale == "1":
                        resized_inputs = inputs
                    else:
                        resized_inputs = F.interpolate(
                                inputs,
                                scale_factor=1 / int(scale),
                                mode="bilinear",
                                align_corners=False
                            )
                    logits.append(disc(resized_inputs).reshape(inputs.size(0), -1))
                logits = torch.cat(logits, dim=1)
                return logits

        self.discriminator = MultiscaleDisc(disc_scales)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else nn.BCEWithLogitsLoss()

        self.detection_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.decoding_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.lecam_regularization_weight = lecam_regularization_weight
        self.lecam_ema_decay = 0.999
        if self.lecam_regularization_weight > 0.0:
            self.ema_real_logits_mean = 0
            self.ema_fake_logits_mean = 0

    @torch.no_grad()
    def calculate_adaptive_weights(
        self,
        losses,
        weights,
        last_layer,
        total_norm=0,
        choose_norm_idx=-1,
        eps=1e-12
    ) -> list:
        # calculate gradients for each loss
        grads = []

        for loss in losses:
            # allows for the computation of gradients w.r.t. intermediate layers if possible
            try:
                grads.append(torch.autograd.grad(
                    loss, last_layer, retain_graph=True)[0])
            except Exception as e:
                print(f"Error computing gradient: {str(e)}")
                grads.append(torch.zeros_like(last_layer))
        grad_norms = [torch.norm(grad) for grad in grads]

        # calculate base weights
        total_weight = sum(weights)
        ratios = [w / total_weight for w in weights]

        # choose total_norm to be the norm of the biggest weight
        assert choose_norm_idx or total_norm > 0, "Either choose_norm_idx or total_norm should be provided"
        if total_norm <= 0:  # if not provided, use the norm of the chosen weight
            # choose_norm_idx = ratios.index(max(ratios))
            total_norm = grad_norms[choose_norm_idx]

        # calculate adaptive weights
        scales = [r * total_norm / (eps + norm)
                  for r, norm in zip(ratios, grad_norms)]
        return scales

    def forward(self,
        inputs: torch.Tensor, reconstructions: torch.Tensor,
        masks: torch.Tensor, msgs: torch.Tensor, preds: torch.Tensor,
        optimizer_idx: int, global_step: int,
        last_layer=None, cond=None, preds_ori: torch.Tensor = None,
    ) -> tuple:
        
        if optimizer_idx == 0:  # embedder update
            weights = {}
            losses = {}

            # perceptual loss
            if self.percep_weight > 0:
                losses["percep"] = self.perceptual_loss(
                    imgs=inputs.contiguous(),
                    imgs_w=reconstructions.contiguous(),
                ).mean()
                weights["percep"] = self.percep_weight

            # discriminator loss
            if self.disc_weight > 0 and global_step >= self.discriminator_iter_start:

                with freeze_grads(self.discriminator):
                    disc_factor = adopt_weight(1.0, global_step, threshold=self.discriminator_iter_start)
                    logits_fake = self.discriminator(reconstructions.contiguous())
                    losses["disc"] = - logits_fake.mean()
                    weights["disc"] = disc_factor * self.disc_weight

            # detection loss
            if self.detect_weight > 0:
                detection_inputs = torch.cat([
                    preds[:, 0:1],
                    preds_ori[:, 0:1]
                ], dim=0)
                detection_targets = torch.cat([
                    torch.ones_like(preds[:, 0:1]),
                    torch.zeros_like(preds_ori[:, 0:1])],
                    dim=0)
                detection_loss = self.detection_loss(
                    detection_inputs.contiguous(),
                    detection_targets.contiguous(),
                ).mean()
                losses["detect"] = detection_loss
                weights["detect"] = self.detect_weight

            # decoding loss
            if self.decode_weight > 0:
                msg_preds = preds[:, 1:]  # b nbits ...
                if msg_preds.dim() == 2:  # extract predicts msg
                    decoding_loss = self.decoding_loss(
                        msg_preds.contiguous(),  # b nbits
                        msgs.contiguous().float()
                    ).mean()
                else:  # extract predicts msg per pixel
                    masks = masks.expand_as(msg_preds).bool()  # b nbits h w
                    bsz, nbits, h, w = msg_preds.size()
                    # b nbits h w
                    msg_targs = msgs.unsqueeze(
                        -1).unsqueeze(-1).expand_as(msg_preds)
                    msg_preds_ = msg_preds.masked_select(masks).view(
                        bsz, nbits, -1)  # b 1 h w -> b nbits n
                    msg_targs_ = msg_targs.masked_select(masks).view(
                        bsz, nbits, -1)  # b 1 h w -> b nbits n
                    decoding_loss = self.decoding_loss(
                        msg_preds_.contiguous(),
                        msg_targs_.contiguous().float()
                    ).mean()
                losses["decode"] = decoding_loss
                weights["decode"] = self.decode_weight

            # calculate adaptive weights
            # turn off adaptive weights if any of the detector or embedder losses are turned off
            if last_layer is not None and self.balanced:
                scales = self.calculate_adaptive_weights(
                    losses=losses.values(),
                    weights=weights.values(),
                    last_layer=last_layer,
                    total_norm=self.total_norm,
                )
                scales = {k: v for k, v in zip(weights.keys(), scales)}
            else:
                scales = weights
            total_loss = sum(scales[key] * losses[key] for key in losses)
            # log
            log = {
                "total_loss": total_loss.clone().detach().mean(),
                **{f"loss_{k}": v.clone().detach().mean() for k, v in losses.items()},
                **{f"scale_{k}": v for k, v in scales.items()}
            }
            return total_loss, log

        if optimizer_idx == 1:  # discriminator update
            if cond is None:
                # detach here prevents gradient leakage to any module other than the discriminator
                logits_real = self.discriminator(
                    inputs.contiguous().detach())
                logits_fake = self.discriminator(
                    reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(
                1.0, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            
            lecam_loss = torch.zeros((), device=inputs.device)
            if self.lecam_regularization_weight > 0.0:
                self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
                self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
                lecam_loss = compute_lecam_loss(
                    torch.mean(logits_real),
                    torch.mean(logits_fake),
                    self.ema_real_logits_mean,
                    self.ema_fake_logits_mean
                ) * self.lecam_regularization_weight

            d_loss += disc_factor * lecam_loss

            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "disc_factor": disc_factor,
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log

    def to(self, device, *args, **kwargs):
        """
        Override for custom perceptual loss to device.
        """
        super().to(device)
        self.perceptual_loss = self.perceptual_loss.to(device)
        return self
