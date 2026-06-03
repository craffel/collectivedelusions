import torch
from torchvision.models import resnet18
import torch.nn as nn
import os
from merge import merge_models

device = torch.device("cpu")
datasets = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

base_model = resnet18(weights=None)
base_model.fc = nn.Linear(base_model.fc.in_features, 10)
base_state_dict = base_model.state_dict()

task_state_dicts = []
for d in datasets:
    ckpt_path = os.path.join("./checkpoints", f"{d}_seed42.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    task_state_dicts.append(ckpt['state_dict'])

# Merge using standard OrthoMerge
print("Merging with orthomerge...")
merged_sd = merge_models(base_state_dict, task_state_dicts, method="orthomerge")

print("\n--- Inspecting Merged State Dict ---")
has_nan = False
has_inf = False
large_val_count = 0

for k, v in merged_sd.items():
    if torch.isnan(v).any():
        print(f"NaN found in {k}!")
        has_nan = True
    if torch.isinf(v).any():
        print(f"Inf found in {k}!")
        has_inf = True
    max_val = v.abs().max().item()
    mean_val = v.abs().mean().item()
    std_val = v.std().item() if v.numel() > 1 else 0.0
    if max_val > 10.0:
        print(f"{k} has extremely large max value: {max_val:.4f} | mean: {mean_val:.4f} | std: {std_val:.4f}")
        large_val_count += 1

print(f"\nNaN status: {has_nan} | Inf status: {has_inf} | Extremely large values in {large_val_count} keys")
