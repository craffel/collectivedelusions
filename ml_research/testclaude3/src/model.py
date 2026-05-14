"""CLIP ViT-B/32 with frozen zero-shot text classifier heads.

Following Ilharco et al. (2023) and Yadav et al. (2023), we fine-tune only the
visual encoder. The classifier head is the cosine similarity between the visual
features and the CLIP text embeddings of class-name prompts -- frozen for all tasks.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, List

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_NAME = os.environ.get("CLIP_ARCH", "ViT-B-32-quickgelu")
PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")


def load_clip(device: str | torch.device = "cpu", arch: str | None = None,
              pretrained: str | None = None):
    # quickgelu variant matches OpenAI's training activation; avoids quality regression.
    arch = arch or MODEL_NAME
    pretrained = pretrained or PRETRAINED
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(arch)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def build_text_classifier(classnames: List[str], prompt_template: str, tokenizer,
                          clip_model, device) -> torch.Tensor:
    """Return [C, D] text embeddings for the classifier head."""
    prompts = [prompt_template.format(c) for c in classnames]
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [C, D]


class CLIPVisualClassifier(nn.Module):
    """CLIP visual encoder + frozen text-embedding classifier head."""

    def __init__(self, clip_model, text_features: torch.Tensor, logit_scale: float | None = None):
        super().__init__()
        # We only need the visual sub-module.
        self.visual = clip_model.visual
        # Logit scale (temperature) -- frozen at CLIP's learned value.
        scale = float(clip_model.logit_scale.exp().item()) if logit_scale is None else logit_scale
        self.register_buffer("logit_scale", torch.tensor(scale))
        # Frozen text features [C, D]
        self.register_buffer("text_features", text_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.visual(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = self.logit_scale * feats @ self.text_features.T
        return logits


def get_visual_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return the visual encoder state dict (the part being merged).

    The model is expected to be a `CLIPVisualClassifier`; we return only the
    `visual.*` parameters so merging operates only on the encoder.
    """
    sd = model.state_dict()
    return {k: v.detach().clone() for k, v in sd.items() if k.startswith("visual.")}


def set_visual_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor]):
    """Load a visual-encoder state dict into a CLIPVisualClassifier in-place."""
    cur = model.state_dict()
    missing = []
    for k, v in sd.items():
        if k in cur:
            cur[k] = v
        else:
            missing.append(k)
    if missing:
        raise ValueError(f"keys not in model: {missing[:5]} ...")
    model.load_state_dict(cur, strict=True)
