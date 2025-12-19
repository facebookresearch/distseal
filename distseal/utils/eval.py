# Copyright (c) Meta Platforms, Inc. and affiliates.
from pathlib import Path
import torch
import torch.nn.functional as F

from typing import Dict, Optional

from distseal.utils.cfg import get_config_from_checkpoint, setup_model, get_extractor
from distseal.models.baselines import build_baseline
from distseal.models.extractor import Extractor
from distseal.models.embedder import Embedder

from omnisealbench.utils.detection import get_detection_and_decoded_keys




class OSBDetector:
    
    model: Extractor
    
    def __init__(self, extractor: Extractor, model_name: str = "", device: str = "cpu", detection_bits: int = 16, image_size: int = 256):
        self.model = extractor
        self.model_name = model_name
        self.device = device
        self.detection_bits = detection_bits
        self.image_size = image_size
    
    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: torch.Tensor,
        detection_threshold: float = 0.0,
        message_threshold: float = 0.0,
        detection_bits: int = 16,
    ) -> torch.Tensor:
        """
        Detect and extract watermark from the contents.
        Args:
            contents (torch.Tensor): The input contents to check for watermarks.
            detection_threshold (float): Threshold for detecting the presence of a watermark.
            message_threshold (float): Threshold to convert tensor message to binary message.
            detection_bits (int): Number of bits to use for detection.
        Returns:
            torch.Tensor: A result containing the detection probability and optionally the extracted secret message.
        """
        
        if self.model_name.lower().startswith("cin"):
            img_sizes = (128, 128)
        else:
            img_sizes = (self.image_size, self.image_size)
        
        if contents.shape[-2:] != img_sizes:
            # resize to the appropriate size
            contents = F.interpolate(contents, size=img_sizes, mode="bilinear", align_corners=False, antialias=True)
        
        extracted_bits = self.model(contents)
        extracted_bits = extracted_bits[:, 1:]
        return get_detection_and_decoded_keys(
            extracted_bits,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold,
        )


def build_extractor(ckpt_path: str, detection_bits: int = 16, device: str = "cpu", image_size: int = 256) -> OSBDetector:
    """build an extractor then wrap it with OmnisealBench API"""
    if "baseline" in ckpt_path:
        method = ckpt_path.split('/')[-1]
        model = build_baseline(method)
    # load distseal model card
    elif ckpt_path.startswith('distseal'):
        model = get_extractor(ckpt_path)
        method = "video_wam"
    # load distseal checkpoints
    else:
        config = get_config_from_checkpoint(Path(ckpt_path))
        model = setup_model(config, Path(ckpt_path))
        method = "video_wam"
    
    model.detector.eval().to(device)
    
    return OSBDetector(extractor=model.detector, model_name=method, device=device, image_size=image_size, detection_bits=detection_bits)
