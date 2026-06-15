import timm
import torch

try:
    print("Creating vit_tiny_patch16_224...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    print("Successfully created vit_tiny_patch16_224!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
except Exception as e:
    print(f"Error: {e}")
