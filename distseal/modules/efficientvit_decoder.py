# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type
from efficientvit.models.efficientvit.cls import ClsHead


class EfficientViTDecoder(nn.Module):
    def __init__(
        self,
        *,
        backbone_type: str = 'b0',
        nbits: int = 0,
        sigmoid_output: bool = False,
    ) -> None:
        """
        Predicts masks given an image embedding, using a simple CNN.

        Arguments:
            embed_dim (int): the input channel dimension
            nbits (int): the number of bits to predict (0 for zero-bit)
            activation (nn.Module): the type of activation to use when
            upscaling masks
            upscale_stages (List[int]): the upscaling factors to use
            upscale_type (str): the type of upscaling to use
            sigmoid_output (bool): whether to apply sigmoid to the output
        """
        super().__init__()
        self.nbits = nbits
        if backbone_type == 'b0':
            self.head = ClsHead(
                in_channels=128,
                width_list=[1024, 1280],
                n_classes=self.nbits + 1,
            )
        elif backbone_type == 'b1':
            self.head = ClsHead(
                in_channels=256,
                width_list=[1536, 1600],
                n_classes=self.nbits + 1,
            )
        elif backbone_type in ['l0', 'l1']:
            self.head = ClsHead(
                in_channels=512,
                width_list=[3072, 3200],
                act_func="gelu",
                n_classes=self.nbits + 1,
            )
        self.sigmoid_output = sigmoid_output
            
    def forward(
        self,
        image_embeddings_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
            image_embeddings (torch.Tensor): the embeddings from the image encoder

        Returns:
            torch.Tensor: batched predicted masks (1+nbits)
        """
        preds = self.head(image_embeddings_dict)

        # Apply sigmoid if needed and return
        if self.sigmoid_output: 
            return F.sigmoid(preds)
        return preds

