"""CLIP model wrappers: visual encoder + frozen text classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


def load_clip(model_name="ViT-B-32", pretrained="openai", device="cuda"):
    """Returns (model, preprocess_train, preprocess_val, tokenizer)."""
    # Use quickgelu variant for OpenAI weights to match their training.
    mn = model_name
    if pretrained == "openai" and not model_name.endswith("-quickgelu"):
        mn = model_name + "-quickgelu"
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(mn, pretrained=pretrained)
    except Exception:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    return model, preprocess, tokenizer


@torch.no_grad()
def compute_text_classifier(model, tokenizer, classnames, template="a photo of a {}", device="cuda"):
    """Return tensor [num_classes, emb_dim] of normalized text embeddings."""
    model.eval()
    tok = tokenizer([template.format(c) for c in classnames]).to(device)
    emb = model.encode_text(tok)
    emb = F.normalize(emb, dim=-1)
    return emb  # [C, D]


class ClipClassifier(nn.Module):
    """Wraps a CLIP visual encoder with a frozen text classifier head."""
    def __init__(self, clip_model, text_classifier, logit_scale=None):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer("text_classifier", text_classifier.detach())
        if logit_scale is None:
            logit_scale = clip_model.logit_scale.detach().exp()
        self.logit_scale = float(logit_scale)

    def forward(self, images):
        feats = self.clip_model.encode_image(images)
        feats = F.normalize(feats, dim=-1)
        logits = self.logit_scale * feats @ self.text_classifier.T
        return logits


def get_visual_state_dict(clip_model):
    """Extract visual encoder state dict (what we fine-tune / merge)."""
    sd = {}
    for k, v in clip_model.state_dict().items():
        if k.startswith("visual."):
            sd[k] = v.detach().cpu().clone()
    return sd


def load_visual_state_dict(clip_model, visual_sd, strict=False):
    """Load visual encoder into a clip_model."""
    full = clip_model.state_dict()
    for k, v in visual_sd.items():
        if k in full:
            full[k] = v.to(full[k].dtype)
    clip_model.load_state_dict(full, strict=False)
    return clip_model
