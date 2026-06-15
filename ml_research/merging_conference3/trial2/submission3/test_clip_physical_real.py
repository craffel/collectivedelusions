import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

# Patch list_repo_templates to prevent Hugging Face Hub 404 errors during tokenizer load
import transformers.tokenization_utils_base
transformers.tokenization_utils_base.list_repo_templates = lambda *args, **kwargs: []

from transformers import CLIPModel, CLIPTokenizer

print("Imported transformers successfully with all HTTPX and hub patches!", flush=True)

# Unsupervised Entropy Loss function for Test-Time Adaptation (TTA)
def calculate_entropy(logits):
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
    return entropy

def get_layer_idx(name):
    if 'vision_model.encoder.layers.' in name:
        parts = name.split('vision_model.encoder.layers.')
        layer_idx = int(parts[1].split('.')[0])
        return layer_idx
    return None

if __name__ == '__main__':
    # Initialize CLIP transform
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    print("Loading real CIFAR-10 test dataset...", flush=True)
    ds_cifar = d.CIFAR10(root='data', download=True, train=False, transform=preprocess)

    print("Loading real GTSRB test dataset...", flush=True)
    ds_gtsrb = d.GTSRB(root='data', split='test', download=True, transform=preprocess)

    # Gather a subset of 50 images from each dataset for fast CPU evaluation
    num_samples = 50
    print(f"Selecting {num_samples} evaluation images from CIFAR-10 and GTSRB...", flush=True)
    
    cifar_images = torch.stack([ds_cifar[i][0] for i in range(num_samples)])
    cifar_labels = torch.tensor([ds_cifar[i][1] for i in range(num_samples)])
    
    gtsrb_images = torch.stack([ds_gtsrb[i][0] for i in range(num_samples)])
    gtsrb_labels = torch.tensor([ds_gtsrb[i][1] for i in range(num_samples)])

    print("Loading pre-trained multimodal CLIP foundation models...", flush=True)
    
    # Load standard pre-trained base CLIPModel
    clip_model_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load fine-tuned expert CLIPModels
    clip_model_task1 = CLIPModel.from_pretrained("tanganke/clip-vit-base-patch32_cifar10")
    clip_model_task2 = CLIPModel.from_pretrained("tanganke/clip-vit-base-patch32_gtsrb")
    
    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    print("All models and tokenizer loaded successfully!", flush=True)
    
    # Define class names and templates
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar10_prompts = [f"a photo of a {c}." for c in cifar10_classes]
    
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
    gtsrb_prompts = [f"a zoomed in photo of a \"{c}\" traffic sign." for c in gtsrb_classes]
    
    # Compute real, pre-trained CLIP text embeddings
    print("Computing real text embeddings for CIFAR-10 and GTSRB prompts...", flush=True)
    inputs_cifar = tokenizer(cifar10_prompts, return_tensors="pt", padding=True)
    inputs_gtsrb = tokenizer(gtsrb_prompts, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        cifar_text_features = clip_model_base.get_text_features(**inputs_cifar)
        cifar_text_features = cifar_text_features / cifar_text_features.norm(dim=-1, keepdim=True)
        
        gtsrb_text_features = clip_model_base.get_text_features(**inputs_gtsrb)
        gtsrb_text_features = gtsrb_text_features / gtsrb_text_features.norm(dim=-1, keepdim=True)
        
    print(f"CIFAR-10 text embeddings: {cifar_text_features.shape}", flush=True)
    print(f"GTSRB text embeddings: {gtsrb_text_features.shape}", flush=True)
    
    # Extract vision parameter dicts
    base_params = {k: v for k, v in clip_model_base.named_parameters()}
    t1_params = {k: v for k, v in clip_model_task1.named_parameters()}
    t2_params = {k: v for k, v in clip_model_task2.named_parameters()}
    
    L = 12
    l_idx = torch.arange(L, dtype=torch.float32) / (L - 1)
    
    print("\nRunning TTA optimization sweeps with real multimodal CLIP classification...", flush=True)
    
    configs = ['task_arithmetic', 'unconstrained', 'unconstrained_tv', 'poly_d2', 'poly_d4', 'splinemerge_const']
    results = {}
    
    for method in configs:
        print(f"\nMethod: {method}", flush=True)
        
        # Initialize coefficients
        if method in ['unconstrained', 'unconstrained_tv']:
            params_t1 = torch.ones(L) * 0.5
            params_t2 = torch.ones(L) * 0.5
            params_t1 = params_t1.detach().requires_grad_(True)
            params_t2 = params_t2.detach().requires_grad_(True)
            optimizer = optim.Adam([params_t1, params_t2], lr=0.02)
        elif method == 'poly_d2':
            params_t1 = torch.zeros(3)
            params_t2 = torch.zeros(3)
            with torch.no_grad():
                params_t1[0] = 0.5
                params_t2[0] = 0.5
            params_t1 = params_t1.detach().requires_grad_(True)
            params_t2 = params_t2.detach().requires_grad_(True)
            optimizer = optim.Adam([params_t1, params_t2], lr=0.02)
        elif method == 'poly_d4':
            params_t1 = torch.zeros(5)
            params_t2 = torch.zeros(5)
            with torch.no_grad():
                params_t1[0] = 0.5
                params_t2[0] = 0.5
            params_t1 = params_t1.detach().requires_grad_(True)
            params_t2 = params_t2.detach().requires_grad_(True)
            optimizer = optim.Adam([params_t1, params_t2], lr=0.02)
        elif method == 'splinemerge_const':
            params_t1 = torch.ones(3) * 0.5
            params_t2 = torch.ones(3) * 0.5
            params_t1 = params_t1.detach().requires_grad_(True)
            params_t2 = params_t2.detach().requires_grad_(True)
            optimizer = optim.Adam([params_t1, params_t2], lr=0.02)
        else:
            params_t1 = torch.ones(L) * 0.5
            params_t2 = torch.ones(L) * 0.5
            
        num_steps = 15 if method != 'task_arithmetic' else 0
        
        # Shell model for functional call
        clip_model_shell = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        history = []
        
        # Evaluate step 0 before optimization starts (for all methods)
        with torch.no_grad():
            # Synthesize initial coefficients
            init_l1 = torch.ones(L) * 0.5
            init_l2 = torch.ones(L) * 0.5
                
            merged_params = {}
            for name in base_params.keys():
                if 'vision_model' in name:
                    idx = get_layer_idx(name)
                    if idx is not None:
                        coef1 = init_l1[idx]
                        coef2 = init_l2[idx]
                        merged_params[name] = base_params[name] + coef1 * (t1_params[name] - base_params[name]) + coef2 * (t2_params[name] - base_params[name])
                    else:
                        merged_params[name] = base_params[name] + 0.5 * (t1_params[name] - base_params[name]) + 0.5 * (t2_params[name] - base_params[name])
                else:
                    merged_params[name] = base_params[name]
                    
            # Split parameters
            merged_params_vision = {}
            merged_params_proj = {}
            for name, param in merged_params.items():
                if name.startswith('vision_model.'):
                    short_name = name.replace('vision_model.', '')
                    merged_params_vision[short_name] = param
                elif name.startswith('visual_projection.'):
                    short_name = name.replace('visual_projection.', '')
                    merged_params_proj[short_name] = param
                    
            # Differentiable forward pass for CIFAR-10
            c_vision_outputs = torch.func.functional_call(clip_model_shell.vision_model, merged_params_vision, args=(cifar_images,))
            c_pooled = c_vision_outputs[1]
            c_image_features = torch.func.functional_call(clip_model_shell.visual_projection, merged_params_proj, args=(c_pooled,))
            c_image_features = c_image_features / c_image_features.norm(dim=-1, keepdim=True)
            
            c_logit_scale = clip_model_shell.logit_scale.exp()
            c_logits = c_image_features @ cifar_text_features.T * c_logit_scale
            c_entropy = calculate_entropy(c_logits)
            c_preds = c_logits.argmax(dim=1)
            c_acc = (c_preds == cifar_labels).float().mean().item() * 100.0
            
            # Differentiable forward pass for GTSRB
            g_vision_outputs = torch.func.functional_call(clip_model_shell.vision_model, merged_params_vision, args=(gtsrb_images,))
            g_pooled = g_vision_outputs[1]
            g_image_features = torch.func.functional_call(clip_model_shell.visual_projection, merged_params_proj, args=(g_pooled,))
            g_image_features = g_image_features / g_image_features.norm(dim=-1, keepdim=True)
            
            g_logit_scale = clip_model_shell.logit_scale.exp()
            g_logits = g_image_features @ gtsrb_text_features.T * g_logit_scale
            g_entropy = calculate_entropy(g_logits)
            g_preds = g_logits.argmax(dim=1)
            g_acc = (g_preds == gtsrb_labels).float().mean().item() * 100.0
            
            avg_acc = 0.5 * (c_acc + g_acc)
            print(f"Step  0 | CIFAR-10 Acc: {c_acc:5.2f}% (Ent: {c_entropy.item():.4f}) | GTSRB Acc: {g_acc:5.2f}% (Ent: {g_entropy.item():.4f}) | Avg Acc: {avg_acc:5.2f}%", flush=True)
            history.append({
                'step': 0,
                'cifar_acc': c_acc,
                'cifar_entropy': c_entropy.item(),
                'gtsrb_acc': g_acc,
                'gtsrb_entropy': g_entropy.item(),
                'avg_acc': avg_acc
            })
            
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Synthesize coefficients
            if method in ['unconstrained', 'unconstrained_tv']:
                l1 = torch.clamp(params_t1, 0.0, 1.0)
                l2 = torch.clamp(params_t2, 0.0, 1.0)
            elif method == 'poly_d2':
                l1 = torch.zeros(L)
                l2 = torch.zeros(L)
                for d in range(3):
                    l1 += params_t1[d] * (l_idx ** d)
                    l2 += params_t2[d] * (l_idx ** d)
                l1 = torch.clamp(l1, 0.0, 1.0)
                l2 = torch.clamp(l2, 0.0, 1.0)
            elif method == 'poly_d4':
                l1 = torch.zeros(L)
                l2 = torch.zeros(L)
                for d in range(5):
                    l1 += params_t1[d] * (l_idx ** d)
                    l2 += params_t2[d] * (l_idx ** d)
                l1 = torch.clamp(l1, 0.0, 1.0)
                l2 = torch.clamp(l2, 0.0, 1.0)
            elif method == 'splinemerge_const':
                l1 = torch.zeros(L)
                l2 = torch.zeros(L)
                for b in range(3):
                    l1[b*4:(b+1)*4] = params_t1[b]
                    l2[b*4:(b+1)*4] = params_t2[b]
                l1 = torch.clamp(l1, 0.0, 1.0)
                l2 = torch.clamp(l2, 0.0, 1.0)
                
            # Perform differentiable weight merging of CLIP Vision Encoder
            merged_params = {}
            for name in base_params.keys():
                if 'vision_model' in name:
                    idx = get_layer_idx(name)
                    if idx is not None:
                        coef1 = l1[idx]
                        coef2 = l2[idx]
                        merged_params[name] = base_params[name] + coef1 * (t1_params[name] - base_params[name]) + coef2 * (t2_params[name] - base_params[name])
                    else:
                        merged_params[name] = base_params[name] + 0.5 * (t1_params[name] - base_params[name]) + 0.5 * (t2_params[name] - base_params[name])
                else:
                    # Keep text parameters constant
                    merged_params[name] = base_params[name]
                    
            # Extract sub-parameter dicts for vision_model and visual_projection
            merged_params_vision = {}
            merged_params_proj = {}
            for name, param in merged_params.items():
                if name.startswith('vision_model.'):
                    short_name = name.replace('vision_model.', '')
                    merged_params_vision[short_name] = param
                elif name.startswith('visual_projection.'):
                    short_name = name.replace('visual_projection.', '')
                    merged_params_proj[short_name] = param
                    
            # Differentiable forward pass for CIFAR-10
            c_vision_outputs = torch.func.functional_call(clip_model_shell.vision_model, merged_params_vision, args=(cifar_images,))
            c_pooled = c_vision_outputs[1]
            c_image_features = torch.func.functional_call(clip_model_shell.visual_projection, merged_params_proj, args=(c_pooled,))
            c_image_features = c_image_features / c_image_features.norm(dim=-1, keepdim=True)
            
            c_logit_scale = clip_model_shell.logit_scale.exp()
            c_logits = c_image_features @ cifar_text_features.T * c_logit_scale
            c_entropy = calculate_entropy(c_logits)
            
            # Differentiable forward pass for GTSRB
            g_vision_outputs = torch.func.functional_call(clip_model_shell.vision_model, merged_params_vision, args=(gtsrb_images,))
            g_pooled = g_vision_outputs[1]
            g_image_features = torch.func.functional_call(clip_model_shell.visual_projection, merged_params_proj, args=(g_pooled,))
            g_image_features = g_image_features / g_image_features.norm(dim=-1, keepdim=True)
            
            g_logit_scale = clip_model_shell.logit_scale.exp()
            g_logits = g_image_features @ gtsrb_text_features.T * g_logit_scale
            g_entropy = calculate_entropy(g_logits)
            
            # Total Unsupervised Entropy Loss on TTA Batch
            loss = c_entropy + g_entropy
            if method == 'unconstrained_tv':
                tv_reg1 = torch.mean((l1[1:] - l1[:-1]) ** 2)
                tv_reg2 = torch.mean((l2[1:] - l2[:-1]) ** 2)
                loss = loss + 5.0 * 0.5 * (tv_reg1 + tv_reg2)
                
            loss.backward()
            optimizer.step()
            
            # Evaluate metrics after optimization step
            with torch.no_grad():
                c_preds = c_logits.argmax(dim=1)
                c_acc = (c_preds == cifar_labels).float().mean().item() * 100.0
                g_preds = g_logits.argmax(dim=1)
                g_acc = (g_preds == gtsrb_labels).float().mean().item() * 100.0
                avg_acc = 0.5 * (c_acc + g_acc)
                print(f"Step {step+1:2d} | CIFAR-10 Acc: {c_acc:5.2f}% (Ent: {c_entropy.item():.4f}) | GTSRB Acc: {g_acc:5.2f}% (Ent: {g_entropy.item():.4f}) | Avg Acc: {avg_acc:5.2f}%", flush=True)
                history.append({
                    'step': step + 1,
                    'cifar_acc': c_acc,
                    'cifar_entropy': c_entropy.item(),
                    'gtsrb_acc': g_acc,
                    'gtsrb_entropy': g_entropy.item(),
                    'avg_acc': avg_acc
                })
            
        with torch.no_grad():
            if method in ['unconstrained', 'unconstrained_tv']:
                l1 = torch.clamp(params_t1, 0.0, 1.0)
                l2 = torch.clamp(params_t2, 0.0, 1.0)
            elif method == 'poly_d2':
                l1 = torch.zeros(L)
                l2 = torch.zeros(L)
                for d in range(3):
                    l1 += params_t1[d] * (l_idx ** d)
                    l2 += params_t2[d] * (l_idx ** d)
                l1 = torch.clamp(l1, 0.0, 1.0)
                l2 = torch.clamp(l2, 0.0, 1.0)
            elif method == 'poly_d4':
                l1 = torch.zeros(L)
                l2 = torch.zeros(L)
                for d in range(5):
                    l1 += params_t1[d] * (l_idx ** d)
                    l2 += params_t2[d] * (l_idx ** d)
                l1 = torch.clamp(l1, 0.0, 1.0)
                l2 = torch.clamp(l2, 0.0, 1.0)
            elif method == 'splinemerge_const':
                l1 = torch.zeros(L)
                l2 = torch.zeros(L)
                for b in range(3):
                    l1[b*4:(b+1)*4] = params_t1[b]
                    l2[b*4:(b+1)*4] = params_t2[b]
                l1 = torch.clamp(l1, 0.0, 1.0)
                l2 = torch.clamp(l2, 0.0, 1.0)
            else:
                l1 = params_t1
                l2 = params_t2
                
            rough1 = torch.mean((l1[1:] - l1[:-1]) ** 2).item()
            rough2 = torch.mean((l2[1:] - l2[:-1]) ** 2).item()
            avg_rough = 0.5 * (rough1 + rough2)
            
            print(f"Completed! Final Roughness (TV): {avg_rough:.6f}", flush=True)
            results[method] = {
                'roughness': avg_rough,
                'l1': l1.numpy().tolist(),
                'l2': l2.numpy().tolist(),
                'history': history
            }
            
    with open("results/clip_physical_real_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nReal CLIP Physical validation completed and saved to results/clip_physical_real_metrics.json!", flush=True)
