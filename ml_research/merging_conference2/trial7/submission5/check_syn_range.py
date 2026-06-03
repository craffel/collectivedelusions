import torch

syn_datasets = torch.load("./checkpoints/synthetic_data.pt", map_location="cpu")
for task, data in syn_datasets.items():
    print(f"Task: {task}")
    print(f"  Shape: {data.shape}")
    print(f"  Mean: {data.mean().item():.4f}")
    print(f"  Std: {data.std().item():.4f}")
    print(f"  Min: {data.min().item():.4f}")
    print(f"  Max: {data.max().item():.4f}")
