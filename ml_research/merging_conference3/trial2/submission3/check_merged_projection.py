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
    
    # Extract parameter dicts
    base_params = {k: v for k, v in clip_model_base.named_parameters()}
    t1_params = {k: v for k, v in clip_model_task1.named_parameters()}
    t2_params = {k: v for k, v in clip_model_task2.named_parameters()}
    
    # Merge vision encoders using 0.5 task arithmetic
    merged_params_vision = {}
    for name in base_params.keys():
        if 'vision_model' in name:
            merged_params_vision[name.replace('vision_model.', '')] = base_params[name] + 0.5 * (t1_params[name] - base_params[name]) + 0.5 * (t2_params[name] - base_params[name])
            
    # Base visual projection is kept frozen
    base_proj_params = {k.replace('visual_projection.', ''): v for k, v in base_params.items() if 'visual_projection' in k}
    
    scale = clip_model_base.logit_scale.exp()
    
    # Differentiable forward pass for CIFAR-10 on merged model
    c_vision_outputs = torch.func.functional_call(clip_model_base.vision_model, merged_params_vision, args=(c_imgs,))
    c_pooled = c_vision_outputs[1]
    c_image_features = torch.func.functional_call(clip_model_base.visual_projection, base_proj_params, args=(c_pooled,))
    c_image_features = c_image_features / c_image_features.norm(dim=-1, keepdim=True)
    logits_c = c_image_features @ text_feat_base_c.T * scale
    acc_c = (logits_c.argmax(dim=1) == c_lbls).float().mean().item() * 100.0
    
    # Differentiable forward pass for GTSRB on merged model
    g_vision_outputs = torch.func.functional_call(clip_model_base.vision_model, merged_params_vision, args=(g_imgs,))
    g_pooled = g_vision_outputs[1]
    g_image_features = torch.func.functional_call(clip_model_base.visual_projection, base_proj_params, args=(g_pooled,))
    g_image_features = g_image_features / g_image_features.norm(dim=-1, keepdim=True)
    logits_g = g_image_features @ text_feat_base_g.T * scale
    acc_g = (logits_g.argmax(dim=1) == g_lbls).float().mean().item() * 100.0
    
    avg_acc = 0.5 * (acc_c + acc_g)
    print(f"Merged model (Task Arithmetic 0.5, 0.5) CIFAR-10 Acc: {acc_c:.2f}%, GTSRB Acc: {acc_g:.2f}%, Avg Acc: {avg_acc:.2f}%")
