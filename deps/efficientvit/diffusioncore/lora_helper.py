"""lora_mingpt.py

Minimal, generic LoRA support for PyTorch models.

This file introduces a LoRALinear module that can transparently replace any
nn.Linear layer in an existing nn.Module with a low‑rank adaptation and the helper
utilities that perform the conversion, (un)merging of LoRA weights, and
freezing of non‑LoRA parameters.

No external LoRA/PEFT dependencies are required – everything is implemented
from scratch. Simply import `apply_lora_to_linear` and call
it on an instantiated model before starting training.

Example
-------
>>> from lora_mingpt import apply_lora_to_linear, mark_only_lora_as_trainable
>>> model = MyModel(...)
>>> apply_lora_to_linear(model, r=8, target_linear_modules=("attn", "mlp"))
>>> mark_only_lora_as_trainable(model)
>>> trainer.fit(model, ...)

When finished you can optionally `merge_lora_weights(model)` to bake the LoRA
corrections into the base model for inference‑only deployments.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LoRALinear",
    "apply_lora_to_linear",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "mark_only_lora_as_trainable",
]


class LoRALinear(nn.Module):
    r"""
    A drop-in replacement for :class:`torch.nn.Linear` that adds a trainable
    Low-Rank Adaptation (LoRA) branch on top of the *frozen* pretrained weight.

    Notes
    -----
    * Mathematically, the layer computes::

          y = x @ Wᵀ + b                      (frozen "base" projection)
              + (x @ Aᵀ) @ Bᵀ * α / r         (low-rank update, trainable)

      where **A** ∈ ℝ^{r×in}, **B** ∈ ℝ^{out×r}, ``r`` ≪ min(in, out) and
      α is a scaling coefficient.

    * The base weight/bias are *not copied* — the wrapper keeps references to
      the original tensors, so no extra memory is used.

    * :py:meth:`merge` can be called after fine-tuning to bake the LoRA update
      into the frozen weight for inference-only checkpoints; the operation is
      exactly reversible via :py:meth:`unmerge`.

    Parameters
    ----------
    base : nn.Linear
        The existing linear layer to wrap.  Its ``weight`` (and ``bias`` if
        present) become the frozen projection.
    r : int, default ``4``
        Rank of the low-rank matrices :math:`A` and :math:`B`.  Setting
        ``r=0`` disables LoRA and the module behaves identically to the
        original :class:`nn.Linear`.
    lora_alpha : int, default ``16``
        Scaling coefficient :math:`α`.  The LoRA update is multiplied by
        ``alpha / r``.
    lora_dropout : float, default ``0.0``
        Dropout probability applied **only** to the input of the LoRA branch
        during training (as described in the paper).

    Attributes
    ----------
    weight : torch.nn.Parameter
        Frozen reference to ``base.weight``.
    bias : torch.nn.Parameter or None
        Frozen reference to ``base.bias`` (if any).
    lora_A, lora_B : torch.nn.Parameter or None
        The trainable low-rank matrices.  Present only when ``r > 0``.
    merge_weights : bool
        Flag that indicates whether the LoRA update has been merged into
        :pyattr:`weight` via :py:meth:`merge`.

    Methods
    -------
    forward(x)
        Apply the frozen projection plus (optionally) the LoRA correction.
    merge()
        Permanently add the LoRA update to :pyattr:`weight`
        and set :pyattr:`merge_weights = True`.
    unmerge()
        Subtract the LoRA update from :pyattr:`weight`
        and set :pyattr:`merge_weights = False`.
    """
    def __init__(
        self,
        base: nn.Linear,           # 🟢 pass the layer you are wrapping
        r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        # ---------------------------------------------------------------
        # 1. Keep the original (frozen) weight & bias *as-is*
        # ---------------------------------------------------------------
        self.weight = base.weight                       # same tensor, same device
        self.weight.requires_grad = False
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = base.bias
            self.bias.requires_grad = False

        # ---------------------------------------------------------------
        # 2. LoRA branch – allocate on *the same* device / dtype
        # ---------------------------------------------------------------
        self.r = int(r)
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.merge_weights = False  # flag to indicate merged/unmerged state        
        
        if r > 0:
            # use weight.new_* to inherit device & dtype
            self.lora_A = nn.Parameter(
                self.weight.new_empty(r, base.in_features)#.normal_(0.0, 0.02)
            )

            # initialise A using the default nn.Linear scheme (Kaiming‑uniform)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            self.lora_B = nn.Parameter(            
                # self.weight.new_zeros(base.out_features, r)    
                self.weight.new_empty(base.out_features, r)
            )
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            self.lora_dropout = (
                nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
            )
        else:                       # r == 0 ⇒ behave exactly like nn.Linear
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assert torch.is_grad_enabled(), "Grad disabled inside LoRA forward()"
        if self.merge_weights or self.r == 0:
            # Fast path when LoRA already merged or disabled
            return F.linear(x, self.weight, self.bias)

        # Standard linear projection from frozen weight
        result = F.linear(x, self.weight, self.bias)

        # Add low‑rank adaptation:  (x @ A^T) @ B^T
        lora_update = F.linear(self.lora_dropout(x), self.lora_A)
        lora_update = F.linear(lora_update, self.lora_B) * self.scaling
        return result + lora_update

    # ------------------------------------------------------------------
    # Utilities for (un)merging – permanently adds LoRA => base weight.
    # This is convenient for inference‑only checkpoints.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def merge(self) -> None:
        """Bake the LoRA correction into the frozen base weight and disable the
        rank‑r branch for faster inference."""
        if self.r == 0 or self.merge_weights:
            return  # nothing to do
        self.weight.add_(self.lora_B @ self.lora_A * self.scaling)
        self.merge_weights = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Reverse ``merge`` – useful if you wish to continue training."""
        if self.r == 0 or not self.merge_weights:
            return
        self.weight.sub_(self.lora_B @ self.lora_A * self.scaling)
        self.merge_weights = False

    # ------------------------------------------------------------------
    # State-dict hook customizes which parameters are saved for this module.
    #
    # - By default, all LoRA and base parameters are saved.
    # - If weights have been merged (after calling merge()), only the merged
    #   weight and bias are saved; LoRA parameters (lora_A, lora_B, etc.) are
    #   excluded from the state dict. This makes the checkpoint compatible with
    #   standard nn.Linear layers and removes any LoRA-specific keys.    
    # - Pass ``strict=False`` when loading into a plain nn.Linear module to ignore extra keys.
    # ------------------------------------------------------------------
    def _save_to_state_dict(self, destination, prefix, keep_vars):  # noqa: D401
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # If weights have been merged, store the *unmerged* variant for
        # reproducibility; this avoids double‑counting when re‑loading.
        if self.merge_weights and self.r > 0 and False:
            destination[prefix + "weight"] = (
                destination[prefix + "weight"] - self.lora_B @ self.lora_A * self.scaling
            )

        # If weights have been merged, only save weight and bias, skip LoRA params
        if self.merge_weights and self.r > 0:
            keys_to_remove = [
                k for k in destination.keys() if k.startswith(prefix + "lora_")
            ]
            for k in keys_to_remove:
                del destination[k]            


# ---------------------------------------------------------------------------
# Helper functions to transparently patch an existing GPT model
# ---------------------------------------------------------------------------

def _find_parent_module(root: nn.Module, dotted_path: str) -> Tuple[nn.Module, str]:
    """Return parent module and last attribute name for a dotted path."""
    parts = dotted_path.split(".")
    for attr in parts[:-1]:
        root = getattr(root, attr)
    return root, parts[-1]


def apply_lora_to_linear(
    model: nn.Module,
    r: int = 4,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_linear_modules: Sequence[str] | None = None,
) -> nn.Module:
    """
    Recursively replace selected ``nn.Linear`` layers in *model* with :class:`LoRALinear`.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model (transformer, diffusion, etc.).
    r, lora_alpha, lora_dropout
        Passed through to :class:`LoRALinear`.
    target_linear_modules : sequence of str, optional
        A list of *substring* patterns. Only linears whose attribute name
        contains one of these patterns will be replaced. Defaults to ("c_attn", "c_proj", "c_fc")
        for backward compatibility. Set to ("",) to patch *all* linears.

    Returns
    -------
    model : nn.Module
        The model with LoRA adapters applied.
    """
    if target_linear_modules is None:
        target_linear_modules = ("c_attn", "c_proj", "c_fc")

    match_all = "" in target_linear_modules           # («»,)  ⇒ patch every Linear        

    for full_name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue                                  # already patched
            
        if isinstance(module, nn.Linear) and (match_all or any(t in full_name for t in target_linear_modules)):
            parent, attr_name = _find_parent_module(model, full_name)
            vanilla: nn.Linear = getattr(parent, attr_name)

            # Construct LoRALinear with same spec and copy params
            lora_mod = LoRALinear(
                vanilla,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            ).to(vanilla.weight.device)          # ⭐ keep device & dtype

            setattr(parent, attr_name, lora_mod)

    return model


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def merge_lora_weights(model: nn.Module) -> None:
    """Recursively ``merge`` all :class:`LoRALinear` children of *model*."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    """Recursively ``unmerge`` all :class:`LoRALinear` children of *model*."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def mark_only_lora_as_trainable(model: nn.Module) -> Iterable[nn.Parameter]:
    """Freeze *all* parameters except those belonging to the LoRA adapters and
    return an iterable of the trainable parameters – handy for your optimizer.
    Works for any model patched with LoRALinear.
    """
    for p in model.parameters():
        p.requires_grad = False
    lora_params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            lora_params.extend([m.lora_A, m.lora_B])
    for p in lora_params:
        p.requires_grad = True
    return lora_params
