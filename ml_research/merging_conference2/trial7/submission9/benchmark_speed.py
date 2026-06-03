import torch
import torch.nn as nn
import time
import os
import copy
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

# Set device to CPU for reproducible execution timing
device = torch.device("cpu")
print(f"Benchmarking calibration speed on device: {device}")

# 1. Load datasets from existing local directories (no downloading needed)
data_dir = "./data"
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading datasets from disk...")
train_mnist = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=transform_gray)
train_fmnist = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform_gray)
train_cifar = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_rgb)

batch_size = 128
train_loaders = {
    "mnist": DataLoader(train_mnist, shuffle=False, batch_size=batch_size),
    "fmnist": DataLoader(train_fmnist, shuffle=False, batch_size=batch_size),
    "cifar10": DataLoader(train_cifar, shuffle=False, batch_size=batch_size)
}

# 2. Load Progenitor and Experts
print("Loading progenitor and expert models...")
progenitor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
progenitor_state = copy.deepcopy(progenitor.state_dict())

def get_expert(task_name):
    expert_path = os.path.join("./experts", f"expert_{task_name}.pt")
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(expert_path, map_location=device))
    return model

experts = {
    "mnist": get_expert("mnist"),
    "fmnist": get_expert("fmnist"),
    "cifar10": get_expert("cifar10")
}

# 3. Import functions directly from run_experiments_v2.py
from run_experiments_v2 import (
    get_standard_merge,
    apply_sp_taac,
    apply_weight_level_ipr,
    apply_bn_level_ipr,
    apply_update_level_ipr,
    apply_spectral_parameter_resonance,
    apply_subspace_aligned_ipr
)

# Create a merged Weight Averaged base model
wa_base = get_standard_merge(experts, progenitor_state, merge_type="WA")

# 4. Measure latency
times = {}

print("\n--- Benchmarking SP-TAAC (Real N=128) ---")
t0 = time.perf_counter()
_ = apply_sp_taac(wa_base, experts, train_loaders, num_samples=128)
times["SP-TAAC (Real N=128)"] = (time.perf_counter() - t0) * 1000.0

print("\n--- Benchmarking W-IPR (Ours, Data-Free) ---")
t0 = time.perf_counter()
_ = apply_weight_level_ipr(wa_base, experts)
times["W-IPR (Ours, Data-Free)"] = (time.perf_counter() - t0) * 1000.0

print("\n--- Benchmarking BN-IPR (Ours, Data-Free) ---")
t0 = time.perf_counter()
_ = apply_bn_level_ipr(wa_base, experts)
times["BN-IPR (Ours, Data-Free)"] = (time.perf_counter() - t0) * 1000.0

print("\n--- Benchmarking U-IPR (Ours, Data-Free) ---")
t0 = time.perf_counter()
_ = apply_update_level_ipr(wa_base, experts, progenitor_state)
times["U-IPR (Ours, Data-Free)"] = (time.perf_counter() - t0) * 1000.0

print("\n--- Benchmarking S-IPR (Ours, Data-Free) ---")
t0 = time.perf_counter()
_ = apply_spectral_parameter_resonance(wa_base, experts, progenitor_state)
times["S-IPR (Ours, Data-Free)"] = (time.perf_counter() - t0) * 1000.0

print("\n--- Benchmarking SA-IPR (Ours, Data-Free, alpha=0.5) ---")
t0 = time.perf_counter()
_ = apply_subspace_aligned_ipr(wa_base, experts, progenitor_state, alpha=0.5)
times["SA-IPR (Ours, alpha=0.5)"] = (time.perf_counter() - t0) * 1000.0

# Print results
print("\n=======================================================")
print("BENCHMARK TIMING RESULTS SUMMARY")
print("=======================================================")
for method, latency in times.items():
    print(f"{method:35s} : {latency:10.2f} ms")
print("=======================================================")
