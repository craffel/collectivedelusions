import importlib.metadata
orig_version = importlib.metadata.version
def patched_version(pkg_name):
    if pkg_name == 'huggingface-hub':
        return '0.35.0'
    return orig_version(pkg_name)
importlib.metadata.version = patched_version

# Apply HTTPX patches
import httpx
orig_head = httpx.Client.head
orig_get = httpx.Client.get

def patched_head(self, url, *args, **kwargs):
    if 'allow_redirects' in kwargs:
        kwargs['follow_redirects'] = kwargs.pop('allow_redirects')
    if 'proxies' in kwargs:
        kwargs.pop('proxies')
    return orig_head(self, url, *args, **kwargs)

def patched_get(self, url, *args, **kwargs):
    if 'allow_redirects' in kwargs:
        kwargs['follow_redirects'] = kwargs.pop('allow_redirects')
    if 'proxies' in kwargs:
        kwargs.pop('proxies')
    return orig_get(self, url, *args, **kwargs)

httpx.Client.head = patched_head
httpx.Client.get = patched_get

# Patch list_repo_templates to prevent Hugging Face Hub 404 errors during tokenizer load
import transformers.tokenization_utils_base
transformers.tokenization_utils_base.list_repo_templates = lambda *args, **kwargs: []

import torch
import torchvision.datasets as d
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer, CLIPVisionModel

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

print("Loading datasets...", flush=True)
ds_cifar = d.CIFAR10(root='data', download=False, train=False, transform=preprocess)
c_imgs = torch.stack([ds_cifar[i][0] for i in range(50)])
c_lbls = torch.tensor([ds_cifar[i][1] for i in range(50)])

ds_gtsrb = d.GTSRB(root='data', split='test', download=False, transform=preprocess)
g_imgs = torch.stack([ds_gtsrb[i][0] for i in range(50)])
g_lbls = torch.tensor([ds_gtsrb[i][1] for i in range(50)])

print("Loading models...", flush=True)
clip_model_base = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_model_task1 = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_cifar10')
clip_model_task2 = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_gtsrb')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar10_prompts = [f'a photo of a {c}.' for c in cifar10_classes]

# We need the 43 GTSRB classes
gtsrb_classes = [
    'red and white circle 20 kph speed limit',
    'red and white circle 30 kph speed limit',
    'red and white circle 50 kph speed limit',
    'red and white circle 60 kph speed limit',
    'red and white circle 70 kph speed limit',
    'red and white circle 80 kph speed limit',
    'end / de-restriction of 80 kph speed limit',
    'red and white circle 100 kph speed limit',
    'red and white circle 120 kph speed limit',
    'red and white circle red car and black car no passing',
    'red and white circle red truck and black car no passing',
    'red and white triangle road intersection warning',
    'white and yellow diamond priority road',
    'red and white upside down triangle yield right-of-way',
    'stop',
    'empty red and white circle',
    'red and white circle no truck entry',
    'red circle with white horizonal stripe no entry',
    'red and white triangle with exclamation mark warning',
    'red and white triangle with black left curve approaching warning',
    'red and white triangle with black right curve approaching warning',
    'red and white triangle with black double curve approaching warning',
    'red and white triangle rough / bumpy road warning',
    'red and white triangle car skidding / slipping warning',
    'red and white triangle with merging / narrow lanes warning',
    'red and white triangle with person digging / construction / road work warning',
    'red and white triangle with traffic light approaching warning',
    'red and white triangle with person walking warning',
    'red and white triangle with child and person walking warning',
    'red and white triangle with bicyle warning',
    'red and white triangle with snowflake / ice warning',
    'red and white triangle with deer warning',
    'white circle with gray strike bar no speed limit',
    'blue circle with white right turn arrow mandatory',
    'blue circle with white left turn arrow mandatory',
    'blue circle with white forward arrow mandatory',
    'blue circle with white forward or right turn arrow mandatory',
    'blue circle with white forward or left turn arrow mandatory',
    'blue circle with white keep right arrow mandatory',
    'blue circle with white keep left arrow mandatory',
    'blue circle with white arrows indicating a traffic circle',
    'white circle with gray strike bar indicating no passing for cars has ended',
    'white circle with gray strike bar indicating no passing for trucks has ended',
]
gtsrb_prompts = [f'a zoomed in photo of a "{c}" traffic sign.' for c in gtsrb_classes]

inputs_c = tokenizer(cifar10_prompts, return_tensors='pt', padding=True)
inputs_g = tokenizer(gtsrb_prompts, return_tensors='pt', padding=True)

with torch.no_grad():
    # Base Text Embeddings from base model
    text_feat_c = clip_model_base.get_text_features(**inputs_c)
    text_feat_c = text_feat_c / text_feat_c.norm(dim=-1, keepdim=True)
    
    text_feat_g = clip_model_base.get_text_features(**inputs_g)
    text_feat_g = text_feat_g / text_feat_g.norm(dim=-1, keepdim=True)
    
    # We extract the base model's visual projection
    base_proj_weight = clip_model_base.visual_projection.weight.data
    
    # Evaluation of CIFAR-10 expert vision encoder with base visual projection
    # Forward pass through expert vision encoder
    # Returns BaseModelOutputWithPooling
    feat_c_pooled = clip_model_task1(c_imgs).pooler_output
    # Project using base projection weight:
    feat_c_projected = feat_c_pooled @ base_proj_weight.T
    feat_c_projected /= feat_c_projected.norm(dim=-1, keepdim=True)
    
    scale = clip_model_base.logit_scale.exp()
    logits_c = feat_c_projected @ text_feat_c.T * scale
    acc_c = (logits_c.argmax(dim=1) == c_lbls).float().mean().item() * 100
    
    # Evaluation of GTSRB expert vision encoder with base visual projection
    feat_g_pooled = clip_model_task2(g_imgs).pooler_output
    # Project using base projection weight:
    feat_g_projected = feat_g_pooled @ base_proj_weight.T
    feat_g_projected /= feat_g_projected.norm(dim=-1, keepdim=True)
    
    logits_g = feat_g_projected @ text_feat_g.T * scale
    acc_g = (logits_g.argmax(dim=1) == g_lbls).float().mean().item() * 100
    
    # Also evaluate base model itself on CIFAR-10 and GTSRB
    base_feat_c_pooled = clip_model_base.vision_model(c_imgs)[1]
    base_feat_c_proj = base_feat_c_pooled @ base_proj_weight.T
    base_feat_c_proj /= base_feat_c_proj.norm(dim=-1, keepdim=True)
    base_logits_c = base_feat_c_proj @ text_feat_c.T * scale
    base_acc_c = (base_logits_c.argmax(dim=1) == c_lbls).float().mean().item() * 100
    
    base_feat_g_pooled = clip_model_base.vision_model(g_imgs)[1]
    base_feat_g_proj = base_feat_g_pooled @ base_proj_weight.T
    base_feat_g_proj /= base_feat_g_proj.norm(dim=-1, keepdim=True)
    base_logits_g = base_feat_g_proj @ text_feat_g.T * scale
    base_acc_g = (base_logits_g.argmax(dim=1) == g_lbls).float().mean().item() * 100
    
    print(f"Base CLIP CIFAR-10 Accuracy: {base_acc_c:.2f}%")
    print(f"Base CLIP GTSRB Accuracy: {base_acc_g:.2f}%")
    print(f"CIFAR-10 Expert Accuracy (with base projection): {acc_c:.2f}%")
    print(f"GTSRB Expert Accuracy (with base projection): {acc_g:.2f}%")
