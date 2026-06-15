import torch
import torch.nn as nn
import timm
import time
import numpy as np

# Set random seed for consistency
torch.manual_seed(42)

print("Initializing ViT-Base Routing Latency Evaluation...")

# 1. Load ViT-Base (pretrained=False to avoid internet download)
device = torch.device("cpu")
vit_base = timm.create_model('vit_base_patch16_224', pretrained=False)
vit_base.eval()

# Helper function to extract Layer 0, Layer 1, and Layer 2 features for ViT-Base
def extract_vit_base_features(model, x, layer=0):
    with torch.no_grad():
        # Layer 0: Patch Embedding
        x_embed = model.patch_embed(x)  # (B, N, D)
        if layer == 0:
            # Spatial average pooling
            return x_embed.mean(dim=1)
        
        # Positional Embedding and prefix tokens
        x_pos = model._pos_embed(x_embed)
        x_drop = model.patch_drop(x_pos)
        x_norm = model.norm_pre(x_drop)
        
        # Layer 1: Block 0
        x1 = model.blocks[0](x_norm)
        if layer == 1:
            # Exclude prefix tokens (cls_token) for pure spatial mean
            return x1[:, 1:, :].mean(dim=1)
            
        # Layer 2: Block 1
        x2 = model.blocks[1](x1)
        if layer == 2:
            return x2[:, 1:, :].mean(dim=1)
            
    return None

# Measure Latency (Single-sample forward, run 100 times to get stable mean)
print("Measuring Single-Sample Processing Latency on CPU for ViT-Base...")

single_x_224 = torch.randn(1, 3, 224, 224)

# PEAR Layer 0
t_start = time.perf_counter()
for _ in range(100):
    _ = extract_vit_base_features(vit_base, single_x_224, layer=0)
t_end = time.perf_counter()
latency_l0 = ((t_end - t_start) / 100) * 1000  # ms

# PEAR Layer 1
t_start = time.perf_counter()
for _ in range(100):
    _ = extract_vit_base_features(vit_base, single_x_224, layer=1)
t_end = time.perf_counter()
latency_l1 = ((t_end - t_start) / 100) * 1000  # ms

# PEAR Layer 2
t_start = time.perf_counter()
for _ in range(100):
    _ = extract_vit_base_features(vit_base, single_x_224, layer=2)
t_end = time.perf_counter()
latency_l2 = ((t_end - t_start) / 100) * 1000  # ms

# Base ViT-Base Full Pass
t_start = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = vit_base(single_x_224)
t_end = time.perf_counter()
latency_vit_full = ((t_end - t_start) / 100) * 1000  # ms

print("\n" + "="*50)
print(f"{'ViT-Base Latency Summary':^50}")
print("="*50)
print(f"PEAR L0 (Patch Embedding) Latency: {latency_l0:.4f} ms")
print(f"PEAR L1 (Block 0 Output) Latency:  {latency_l1:.4f} ms")
print(f"PEAR L2 (Block 1 Output) Latency:  {latency_l2:.4f} ms")
print(f"Base ViT-Base Full Pass Latency:   {latency_vit_full:.4f} ms")
print("-"*50)
print("Routing Compute Overhead relative to Base ViT-Base Full Pass:")
print(f"PEAR L0 Overhead: {latency_l0 / latency_vit_full * 100:.2f}%")
print(f"PEAR L1 Overhead: {latency_l1 / latency_vit_full * 100:.2f}%")
print(f"PEAR L2 Overhead: {latency_l2 / latency_vit_full * 100:.2f}%")
print("="*50)

# Save results to vit_base_results.txt
with open("vit_base_results.txt", "w") as f:
    f.write(f"Latency PEAR L0: {latency_l0:.4f} ms\n")
    f.write(f"Latency PEAR L1: {latency_l1:.4f} ms\n")
    f.write(f"Latency PEAR L2: {latency_l2:.4f} ms\n")
    f.write(f"Latency ViT-Base Full: {latency_vit_full:.4f} ms\n")
    f.write(f"PEAR L0 Overhead: {latency_l0 / latency_vit_full * 100:.2f}%\n")
    f.write(f"PEAR L1 Overhead: {latency_l1 / latency_vit_full * 100:.2f}%\n")
    f.write(f"PEAR L2 Overhead: {latency_l2 / latency_vit_full * 100:.2f}%\n")

print("\nSuccessfully saved results to 'vit_base_results.txt'")
