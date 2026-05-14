"""CLIP ViT-B/32 vision encoder + task heads for model-merging experiments."""
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


MODEL_ID = "openai/clip-vit-base-patch32"


def build_processor() -> CLIPImageProcessor:
    return CLIPImageProcessor.from_pretrained(MODEL_ID)


def build_vision_encoder() -> CLIPVisionModel:
    return CLIPVisionModel.from_pretrained(MODEL_ID)


class ClipClassifier(nn.Module):
    """CLIPVisionModel + a linear head on the pooled output (CLS token after final LN)."""

    HIDDEN = 768  # for ViT-B/32

    def __init__(self, num_classes: int, vision: CLIPVisionModel | None = None):
        super().__init__()
        self.vision = vision if vision is not None else build_vision_encoder()
        self.head = nn.Linear(self.HIDDEN, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision(pixel_values=pixel_values)
        feats = out.pooler_output  # (B, 768)
        return self.head(feats)


# --- helpers used by training and by the merging code ----------------------

def vision_state_dict(model: ClipClassifier) -> Dict[str, torch.Tensor]:
    """Return only the CLIPVisionModel parameter tensors (these are what we merge)."""
    return {k: v.detach().cpu().clone() for k, v in model.vision.state_dict().items()}


def head_state_dict(model: ClipClassifier) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}


def load_vision_into(model: ClipClassifier, sd: Dict[str, torch.Tensor]):
    model.vision.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in sd.items()})


# names of linear (nn.Linear) weights inside the vision encoder.
# Useful because RegMean / ACTMat operate per *linear layer*: we treat each linear
# weight matrix as a separate W. We exclude biases, layer norms, embeddings.
def linear_weight_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    keys = []
    for k, v in state_dict.items():
        if not k.endswith(".weight"):
            continue
        if v.ndim != 2:
            continue
        # filter out 1D things that nevertheless have .weight; only matrices.
        keys.append(k)
    return keys
