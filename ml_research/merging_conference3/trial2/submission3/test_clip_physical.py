import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Apply the version patch before importing transformers
import importlib.metadata
orig_version = importlib.metadata.version
def patched_version(pkg_name):
    if pkg_name == 'huggingface-hub':
        return '0.35.0'
    return orig_version(pkg_name)
importlib.metadata.version = patched_version

# Apply the httpx monkey-patches for compatibility between newer httpx and transformers session client
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

from transformers import CLIPVisionModel

print("Successfully imported transformers and applied HTTPX compatibility patches!", flush=True)

# Unsupervised Entropy Loss function for Test-Time Adaptation (TTA)
def calculate_entropy(logits):
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
    return entropy

def get_layer_idx(name):
    # Parameter names like: 'vision_model.encoder.layers.0.self_attn.q_proj.weight'
    if 'layers.' in name:
        parts = name.split('layers.')
        layer_idx = int(parts[1].split('.')[0])
        return layer_idx
    return None

if __name__ == '__main__':
    print("Loading pre-trained foundation models from Hugging Face...", flush=True)
    
    print("Loading clip_model_base...", flush=True)
    clip_model_base = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading clip_model_task1 (cifar10)...", flush=True)
    clip_model_task1 = CLIPVisionModel.from_pretrained("tanganke/clip-vit-base-patch32_cifar10")
    
    print("Loading clip_model_task2 (gtsrb)...", flush=True)
    clip_model_task2 = CLIPVisionModel.from_pretrained("tanganke/clip-vit-base-patch32_gtsrb")
    
    print("Models loaded successfully!", flush=True)
    
    # Extract parameter dicts
    base_params = {k: v for k, v in clip_model_base.named_parameters()}
    t1_params = {k: v for k, v in clip_model_task1.named_parameters()}
    t2_params = {k: v for k, v in clip_model_task2.named_parameters()}
    
    # 12 layers in CLIP ViT-B/32
    L = 12
    l_idx = torch.arange(L, dtype=torch.float32) / (L - 1)
    
    # Let's run a test TTA loop on a dummy batch of 4 images
    dummy_images = torch.randn(4, 3, 224, 224)
    
    # Define a fixed random classifier head to project the 768 features to 10 classes
    class_head = nn.Linear(768, 10)
    for p in class_head.parameters():
        p.requires_grad = False
        
    print("\nRunning TTA optimization sweeps on physical CLIP Vision Transformer...", flush=True)
    
    # Run over 3 configurations: Task Arithmetic, Unconstrained, PolyMerge d=2
    configs = ['task_arithmetic', 'unconstrained', 'poly_d2']
    
    results = {}
    
    for method in configs:
        print(f"\nMethod: {method}", flush=True)
        
        # Initialize coefficients
        if method == 'unconstrained':
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
        else:
            # Task Arithmetic
            params_t1 = torch.ones(L) * 0.5
            params_t2 = torch.ones(L) * 0.5
            
        # Run TTA for 10 steps (for quick verification)
        num_steps = 15 if method != 'task_arithmetic' else 0
        
        # We need a shell model to execute functional calls
        clip_model_shell = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Synthesize coefficients
            if method == 'unconstrained':
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
                
            # Perform differentiable weight merging
            merged_params = {}
            for name in base_params.keys():
                idx = get_layer_idx(name)
                if idx is not None:
                    coef1 = l1[idx]
                    coef2 = l2[idx]
                    merged_params[name] = base_params[name] + coef1 * (t1_params[name] - base_params[name]) + coef2 * (t2_params[name] - base_params[name])
                else:
                    merged_params[name] = base_params[name] + 0.5 * (t1_params[name] - base_params[name]) + 0.5 * (t2_params[name] - base_params[name])
                    
            # Differentiable forward pass on CLIP!
            outputs = torch.func.functional_call(clip_model_shell, merged_params, args=(dummy_images,))
            features = outputs.pooler_output
            logits = class_head(features)
            entropy = calculate_entropy(logits)
            
            entropy.backward()
            optimizer.step()
            
            print(f"Step {step+1:2d} | Entropy: {entropy.item():.6f}", flush=True)
            
        # Evaluate final state
        with torch.no_grad():
            if method == 'unconstrained':
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
                'l2': l2.numpy().tolist()
            }
            
    with open("results/clip_physical_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nCLIP Physical validation completed and saved to results/clip_physical_metrics.json!", flush=True)
