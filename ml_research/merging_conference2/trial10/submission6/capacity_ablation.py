import os
import copy
import hashlib
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Transforms
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530])
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Load Test Datasets
print("Loading test datasets...")
test_sets = {
    'mnist': datasets.MNIST(root='./data', train=False, download=False, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='./data', train=False, download=False, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='./data', train=False, download=False, transform=cifar_transform)
}

test_loaders = {
    name: DataLoader(test_sets[name], batch_size=256, shuffle=False, num_workers=4)
    for name in test_sets
}

cal_loaders = {}
for name in test_sets:
    cal_subset = Subset(test_sets[name], range(32))
    cal_loaders[name] = DataLoader(cal_subset, batch_size=32, shuffle=False)

# Progenitor Model Definition
def get_progenitor():
    model = resnet18()
    model.fc = nn.Identity()
    return model

# Load Trained Checkpoints
print("Loading checkpoints...")
progenitor_state = torch.load("checkpoint_progenitor.pth", map_location=device)

expert_states = {
    'mnist': torch.load("checkpoint_mnist.pth", map_location=device),
    'fmnist': torch.load("checkpoint_fmnist.pth", map_location=device),
    'cifar10': torch.load("checkpoint_cifar10.pth", map_location=device)
}

expert_backbones = {}
expert_heads = {}
tasks = ['mnist', 'fmnist', 'cifar10']

for t in tasks:
    backbone_sd = {k[2:]: v for k, v in expert_states[t].items() if k.startswith("0.")}
    head_sd = {k[2:]: v for k, v in expert_states[t].items() if k.startswith("1.")}
    expert_backbones[t] = backbone_sd
    expert_heads[t] = head_sd

def get_phase_key(param_name, param_shape, task_idx, device):
    seed_str = f"task_{task_idx}_{param_name}"
    seed_hash = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest()[:8], 16)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_hash)
    
    signs = torch.randint(0, 2, size=param_shape, generator=gen, device=device) * 2 - 1
    return signs.float()

def evaluate_model(backbone_sd, head_sd, task_name):
    backbone = get_progenitor()
    backbone.load_state_dict(backbone_sd)
    
    head = nn.Linear(512, 10)
    head.load_state_dict(head_sd)
    
    model = nn.Sequential(backbone, head).to(device)
    model.eval()
    
    loader = test_loaders[task_name]
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return correct / total * 100.0

def calibrate_model_bn(backbone_sd, head_sd, task_name):
    backbone = get_progenitor()
    backbone.load_state_dict(backbone_sd)
    
    head = nn.Linear(512, 10)
    head.load_state_dict(head_sd)
    
    model = nn.Sequential(backbone, head).to(device)
    
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = 1.0
            m.reset_running_stats()
            m.train()
            
    with torch.no_grad():
        for inputs, _ in cal_loaders[task_name]:
            inputs = inputs.to(device)
            _ = model(inputs)
            break
            
    model.eval()
    return backbone.state_dict()

# Capacity sweep
print("Starting capacity sweep (K from 3 to 20)...")
capacity_results = {}

for K in [3, 5, 8, 10, 15, 20]:
    print(f"\n--- Sweeping K = {K} ---")
    
    # 1. Build the K-expert merged backbone
    hsa_backbone_sd = {}
    for key in progenitor_state.keys():
        if progenitor_state[key].is_floating_point() and "weight" in key:
            hsa_update = torch.zeros_like(progenitor_state[key])
            
            # The 3 real tasks
            for idx, t in enumerate(tasks):
                tau = expert_backbones[t][key] - progenitor_state[key]
                P = get_phase_key(key, progenitor_state[key].shape, idx, device)
                hsa_update += tau * P
                
            # K-3 synthetic tasks matching the real update standard deviation per parameter
            if K > 3:
                std_layer = torch.stack([expert_backbones[t][key] - progenitor_state[key] for t in tasks]).std(dim=0)
                std_layer = torch.clamp(std_layer, min=1e-8)
                
                for idx in range(3, K):
                    tau_synthetic = torch.randn_like(progenitor_state[key]) * std_layer
                    P = get_phase_key(key, progenitor_state[key].shape, idx, device)
                    hsa_update += tau_synthetic * P
                    
            hsa_backbone_sd[key] = progenitor_state[key] + (hsa_update / (K ** 0.5))
        else:
            hsa_backbone_sd[key] = progenitor_state[key].clone()
            
    # 2. Evaluate performance on the 3 real tasks
    task_accs = {}
    for idx, t in enumerate(tasks):
        retrieved_backbone_sd = {}
        for key in progenitor_state.keys():
            if progenitor_state[key].is_floating_point() and "weight" in key:
                hsa_update_tensor = hsa_backbone_sd[key] - progenitor_state[key]
                P = get_phase_key(key, progenitor_state[key].shape, idx, device)
                retrieved_backbone_sd[key] = progenitor_state[key] + ((K ** 0.5) * hsa_update_tensor * P)
            else:
                retrieved_backbone_sd[key] = hsa_backbone_sd[key].clone()
                
        # Calibrate BN
        cal_backbone = calibrate_model_bn(retrieved_backbone_sd, expert_heads[t], t)
        acc = evaluate_model(cal_backbone, expert_heads[t], t)
        task_accs[t] = acc
        print(f"  Task {t.upper()} Accuracy: {acc:.2f}%")
        
    avg_acc = sum(task_accs.values()) / 3
    print(f"  Average Accuracy: {avg_acc:.2f}%")
    capacity_results[str(K)] = {
        'tasks': task_accs,
        'average': avg_acc
    }

# Save results to capacity_results.json
with open("capacity_results.json", "w") as f:
    json.dump(capacity_results, f, indent=4)
print("\nCapacity sweep finished. Results saved to capacity_results.json")
