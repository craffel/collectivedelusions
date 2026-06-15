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
from transformers import CLIPModel, CLIPTokenizer

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
clip_model_task1 = CLIPModel.from_pretrained('tanganke/clip-vit-base-patch32_cifar10')
clip_model_task2 = CLIPModel.from_pretrained('tanganke/clip-vit-base-patch32_gtsrb')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar10_prompts = [f'a photo of a {c}.' for c in cifar10_classes]

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
    # 1. Base Text Embeddings
    text_feat_base_c = clip_model_base.get_text_features(**inputs_c)
    text_feat_base_c = text_feat_base_c / text_feat_base_c.norm(dim=-1, keepdim=True)
    
    text_feat_base_g = clip_model_base.get_text_features(**inputs_g)
    text_feat_base_g = text_feat_base_g / text_feat_base_g.norm(dim=-1, keepdim=True)
    
    # 2. Expert's own Text Embeddings
    text_feat_exp_c = clip_model_task1.get_text_features(**inputs_c)
    text_feat_exp_c = text_feat_exp_c / text_feat_exp_c.norm(dim=-1, keepdim=True)
    
    text_feat_exp_g = clip_model_task2.get_text_features(**inputs_g)
    text_feat_exp_g = text_feat_exp_g / text_feat_exp_g.norm(dim=-1, keepdim=True)
    
    # Forward pass on task 1 expert
    img_feat_c1 = clip_model_task1.get_image_features(c_imgs)
    img_feat_c1 = img_feat_c1 / img_feat_c1.norm(dim=-1, keepdim=True)
    
    # Forward pass on task 2 expert
    img_feat_g2 = clip_model_task2.get_image_features(g_imgs)
    img_feat_g2 = img_feat_g2 / img_feat_g2.norm(dim=-1, keepdim=True)
    
    scale1 = clip_model_task1.logit_scale.exp()
    scale2 = clip_model_task2.logit_scale.exp()
    
    # Eval CIFAR-10
    logits_c_base = img_feat_c1 @ text_feat_base_c.T * scale1
    logits_c_exp = img_feat_c1 @ text_feat_exp_c.T * scale1
    acc_c_base = (logits_c_base.argmax(dim=1) == c_lbls).float().mean().item() * 100
    acc_c_exp = (logits_c_exp.argmax(dim=1) == c_lbls).float().mean().item() * 100
    
    # Eval GTSRB
    logits_g_base = img_feat_g2 @ text_feat_base_g.T * scale2
    logits_g_exp = img_feat_g2 @ text_feat_exp_g.T * scale2
    acc_g_base = (logits_g_base.argmax(dim=1) == g_lbls).float().mean().item() * 100
    acc_g_exp = (logits_g_exp.argmax(dim=1) == g_lbls).float().mean().item() * 100
    
    print(f"CIFAR-10 accuracy with Base Text Embeddings: {acc_c_base:.2f}%")
    print(f"CIFAR-10 accuracy with Expert Text Embeddings: {acc_c_exp:.2f}%")
    print(f"GTSRB accuracy with Base Text Embeddings: {acc_g_base:.2f}%")
    print(f"GTSRB accuracy with Expert Text Embeddings: {acc_g_exp:.2f}%")
