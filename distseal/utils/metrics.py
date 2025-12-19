# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
To run
    python -m distseal.evals.metrics
"""

import os
import math
import subprocess
import tempfile
import re
import numpy as np
from scipy import stats

import torch
import pytorch_msssim

def psnr(x, y, is_video=False):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
        is_video: If True, the PSNR is computed over the entire batch, not on each image separately
    """
    delta = 255 * (x - y)
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    peak = 20 * math.log10(255.0)
    avg_on_dims = (0,1,2,3) if is_video else (1,2,3)
    noise = torch.mean(delta**2, dim=avg_on_dims)
    psnr = peak - 10*torch.log10(noise)
    return psnr

def ssim(x, y, data_range=1.0):
    """
    Return SSIM
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    return pytorch_msssim.ssim(x, y, data_range=data_range, size_average=False)

def msssim(x, y, data_range=1.0):
    """
    Return MSSSIM
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    return pytorch_msssim.ms_ssim(x, y, data_range=data_range, size_average=False)

def linf(x, y, data_range=1.0):
    """
    Return L_inf in pixel space (integer between 0 and 255)
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    multiplier = 255.0 / data_range
    return torch.max(torch.abs(x - y)) * multiplier
    
def iou(preds, targets, threshold=0.0, label=1):
    """
    Return IoU for a specific label (0 or 1).
    Args:
        preds (torch.Tensor): Predicted masks with shape Bx1xHxW
        targets (torch.Tensor): Target masks with shape Bx1xHxW
        label (int): The label to calculate IoU for (0 for background, 1 for foreground)
        threshold (float): Threshold to convert predictions to binary masks
    """
    preds = preds > threshold  # Bx1xHxW
    targets = targets > 0.5
    if label == 0:
        preds = ~preds
        targets = ~targets
    intersection = (preds & targets).float().sum((1,2,3))  # B
    union = (preds | targets).float().sum((1,2,3))  # B
    # avoid division by zero
    union[union == 0.0] = intersection[union == 0.0] = 1
    iou = intersection / union
    return iou

def accuracy(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Return accuracy
    Args:
        preds (torch.Tensor): Predicted masks with shape Bx1xHxW
        targets (torch.Tensor): Target masks with shape Bx1xHxW
    """
    preds = preds > threshold  # b 1 h w
    targets = targets > 0.5
    correct = (preds == targets).float()  # b 1 h w
    accuracy = torch.mean(correct, dim=(1,2,3))  # b
    return accuracy

def pvalue(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Return p values
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        mask (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
    """
    nbits = targets.shape[-1]
    bit_accs = bit_accuracy(preds, targets, mask, threshold)  # b
    pvalues = [stats.binomtest(int(p*nbits), nbits, 0.5, alternative='greater').pvalue for p in bit_accs]
    return torch.tensor(pvalues)  # b

def plogp(p: torch.Tensor) -> torch.Tensor:
    """
    Return p log p
    Args:
        p (torch.Tensor): Probability tensor with shape BxK
    """
    plogp = p * torch.log2(p)
    plogp[p == 0] = 0
    return plogp

def capacity(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Return normalized bit accuracy, defined as the capacity of the nbits channels,
    in the case of a binary symmetric channel of error probability being the bit. acc.
    """
    nbits = targets.shape[-1]
    bit_accs = bit_accuracy(preds, targets, mask, threshold)  # b
    entropy = - plogp(bit_accs) - plogp(1-bit_accs)
    capacity = 1 - entropy
    capacity = nbits * capacity
    return capacity

def bit_accuracy(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Return bit accuracy
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        mask (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    preds = preds > threshold  # b k ...
    if preds.dim() == 4:  # bit preds are pixelwise
        bsz, nbits, h, w = preds.size()
        if mask is not None:
            mask = mask.expand_as(preds).bool()
            preds = preds.masked_select(mask).view(bsz, nbits, -1)  # b k n
            preds = preds.mean(dim=-1, dtype=float)  # b k
        else:
            preds = preds.mean(dim=(-2, -1), dtype=float) # b k
    preds = preds > 0.5  # b k
    targets = targets > 0.5  # b k
    correct = (preds == targets).float()  # b k
    bit_acc = torch.mean(correct, dim=-1)  # b
    return bit_acc

def bit_accuracy_1msg(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    masks: torch.Tensor = None,
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Computes the bit accuracy for each pixel, then averages over all pixels.
    Better for "k-bit" evaluation during training since it's independent of detection performance.
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    preds = preds > threshold  # b k h w
    targets = targets > 0.5  # b k
    correct = (preds == targets.unsqueeze(-1).unsqueeze(-1)).float()  # b k h w
    if masks is not None:  
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(correct).bool()
        correct_list = [correct[i].masked_select(masks[i]) for i in range(len(masks))]
        bit_acc = torch.tensor([torch.mean(correct_list[i]).item() for i in range(len(correct_list))])
    else:
        bit_acc = torch.mean(correct, dim=(1,2,3))  # b
    return bit_acc

def bit_accuracy_inference(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    masks: torch.Tensor,
    method: str = 'hard',
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Computes the message by averaging over all pixels, then computes the bit accuracy.
    Closer to how the model is evaluated during inference.
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
        method (str): Method to compute bit accuracy. Options: 'hard', 'soft'
    """
    if method == 'hard':
        # convert every pixel prediction to binary, select based on masks, and average
        preds = preds > threshold  # b k h w
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == 'semihard':
        # select every pixel prediction based on masks, and average
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == 'soft':
        # average every pixel prediction, use masks "softly" as weights for averaging
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(preds)  # b k h w
        preds = torch.sum(preds * masks, dim=(2,3)) / torch.sum(masks, dim=(2,3))  # b k
    preds = preds > 0.5  # b k
    targets = targets > 0.5  # b k
    correct = (preds == targets).float()  # b k
    bit_acc = torch.mean(correct, dim=(1))  # b
    return bit_acc

def bit_accuracy_mv(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    masks: torch.Tensor = None,
    threshold: float = 0.0
) -> torch.Tensor:
    """
    (Majority vote)
    Return bit accuracy
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    preds = preds > threshold  # b k h w
    targets = targets > 0.5  # b k
    correct = (preds == targets.unsqueeze(-1).unsqueeze(-1)).float()  # b k h w
    if masks is not None:  
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(correct).bool()
        preds = preds.masked_select(masks).view(bsz, nbits, -1)  # b k n
        # correct = correct.masked_select(masks).view(bsz, nbits, -1)  # b k n
        # correct = correct.unsqueeze(-1)  # b k n 1
    # Perform majority vote for each bit
    preds_majority, _ = torch.mode(preds, dim=-1)  # b k
    # Compute bit accuracy
    correct = (preds_majority == targets).float()  # b k
    # bit_acc = torch.mean(correct, dim=(1,2,3))  # b
    bit_acc = torch.mean(correct, dim=-1)  # b
    return bit_acc


def message_accuracy(message: torch.Tensor, gt_message: torch.Tensor) -> float:
    """Calculate the message accuracy between the detected message and the ground truth message."""

    decoded = 2 * message - 1
    decoded = decoded.to(gt_message.device)  # b k
    
    gt_message = 2 * gt_message - 1
    if gt_message.ndim == 1:
        gt_message = gt_message.repeat(decoded.shape[0], 1)  # b k

    # Bit accuracy: number of matching bits between the key and the message, divided by the total number of bits.
    # p-value: probability of observing a bit accuracy as high as the one observed, assuming the null hypothesis that the image is genuine.
    diff = ~torch.logical_xor(decoded > 0, gt_message > 0)  # b k -> b k

    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
    avg_bit_acc = bit_accs.mean()
    return avg_bit_acc.item()


if __name__ == '__main__':
    # Test the PSNR function
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    print("> test psnr")
    try:
        print("OK!", psnr(x, y))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Test the IoU function
    preds = torch.rand(1, 1, 256, 256)
    targets = torch.rand(1, 1, 256, 256)
    print("> test iou")
    try:
        print("OK!", iou(preds, targets))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Test the accuracy function
    preds = torch.rand(1, 1, 256, 256)
    targets = torch.rand(1, 1, 256, 256)
    print("> test accuracy")
    try:
        print("OK!", accuracy(preds, targets))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
