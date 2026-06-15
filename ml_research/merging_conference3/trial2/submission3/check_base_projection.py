import torch
import torchvision.datasets as d
from torchvision import transforms

# Apply the version patch before importing transformers
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

import transformers.tokenization_utils_base
transformers.tokenization_utils_base.list_repo_templates = lambda *args, **kwargs: []

from transformers import CLIPModel, CLIPTokenizer

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

print("Loading datasets...")
ds_cifar = d.CIFAR10(root='data', download=False, train=False, transform=preprocess)
c_imgs = torch.stack([ds_cifar[i][0] for i in range(50)])
c_lbls = torch.tensor([ds_cifar[i][1] for i in range(50)])

ds_gtsrb = d.GTSRB(root='data', split='test', download=False, transform=preprocess)
g_imgs = torch.stack([ds_gtsrb[i][0] for i in range(50)])
g_lbls = torch.tensor([ds_gtsrb[i][1] for i in range(50)])

print("Loading CLIP Model Base...")
clip_model_base = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
print("Loading CLIP Model Task 1...")
clip_model_task1 = CLIPModel.from_pretrained('tanganke/clip-vit-base-patch32_cifar10')
print("Loading CLIP Model Task 2...")
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
    # Base text features
    text_feat_base_c = clip_model_base.get_text_features(**inputs_c)
    text_feat_base_c = text_feat_base_c / text_feat_base_c.norm(dim=-1, keepdim=True)
    
    text_feat_base_g = clip_model_base.get_text_features(**inputs_g)
    text_feat_base_g = text_feat_base_g / text_feat_base_g.norm(dim=-1, keepdim=True)
    
    # Base Model Visual Projection
    base_visual_projection = clip_model_base.visual_projection
    scale = clip_model_base.logit_scale.exp()
    
    # 1. Base model itself
    img_feat_base_c = clip_model_base.get_image_features(c_imgs)
    img_feat_base_c = img_feat_base_c / img_feat_base_c.norm(dim=-1, keepdim=True)
    logits_base_c = img_feat_base_c @ text_feat_base_c.T * scale
    acc_base_c = (logits_base_c.argmax(dim=1) == c_lbls).float().mean().item() * 100
    
    img_feat_base_g = clip_model_base.get_image_features(g_imgs)
    img_feat_base_g = img_feat_base_g / img_feat_base_g.norm(dim=-1, keepdim=True)
    logits_base_g = img_feat_base_g @ text_feat_base_g.T * scale
    acc_base_g = (logits_base_g.argmax(dim=1) == g_lbls).float().mean().item() * 100
    
    print(f"Base model CIFAR-10 Accuracy: {acc_base_c:.2f}%")
    print(f"Base model GTSRB Accuracy: {acc_base_g:.2f}%")
    
    # 2. Expert task 1 (CIFAR-10) using task1's vision model and base visual projection
    pooled_t1_c = clip_model_task1.vision_model(c_imgs)[1]
    img_feat_t1_c = base_visual_projection(pooled_t1_c)
    img_feat_t1_c = img_feat_t1_c / img_feat_t1_c.norm(dim=-1, keepdim=True)
    logits_t1_c = img_feat_t1_c @ text_feat_base_c.T * scale
    acc_t1_c = (logits_t1_c.argmax(dim=1) == c_lbls).float().mean().item() * 100
    
    print(f"CIFAR-10 Expert (projected by Base Projection) Accuracy: {acc_t1_c:.2f}%")
    
    # 3. Expert task 2 (GTSRB) using task2's vision model and base visual projection
    pooled_t2_g = clip_model_task2.vision_model(g_imgs)[1]
    img_feat_t2_g = base_visual_projection(pooled_t2_g)
    img_feat_t2_g = img_feat_t2_g / img_feat_t2_g.norm(dim=-1, keepdim=True)
    logits_t2_g = img_feat_t2_g @ text_feat_base_g.T * scale
    acc_t2_g = (logits_t2_g.argmax(dim=1) == g_lbls).float().mean().item() * 100
    
    print(f"GTSRB Expert (projected by Base Projection) Accuracy: {acc_t2_g:.2f}%")
