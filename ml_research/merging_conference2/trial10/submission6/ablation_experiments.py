import os
import copy
import hashlib
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# 1. Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

# 2. Data Transforms
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

# 3. Load Test Datasets
print("Loading test datasets...")
test_sets = {
    'mnist': datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='./data', train=False, download=True, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
}

test_loaders = {
    name: DataLoader(test_sets[name], batch_size=256, shuffle=False, num_workers=4)
    for name in test_sets
}

print("Creating calibration subsets (N=32)...")
cal_loaders = {}
for name in test_sets:
    cal_subset = Subset(test_sets[name], range(32))
    cal_loaders[name] = DataLoader(cal_subset, batch_size=32, shuffle=False)

# 4. Progenitor Model Definition
def get_progenitor():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model

# 5. Load Trained Checkpoints
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

# PTQ Implementation
def quantize_tensor(tensor, num_bits=8, per_channel=False):
    if num_bits is None:
        return tensor
    qmax = 2**(num_bits - 1) - 1
    
    if per_channel:
        orig_shape = tensor.shape
        flat_tensor = tensor.view(orig_shape[0], -1)
        max_vals = flat_tensor.abs().max(dim=1, keepdim=True)[0]
        max_vals = torch.clamp(max_vals, min=1e-8)
        delta = max_vals / qmax
        
        q_tensor = torch.round(tensor / delta.view(orig_shape[0], *([1] * (len(orig_shape) - 1))))
        q_tensor = torch.clamp(q_tensor, -qmax, qmax)
        dq_tensor = q_tensor * delta.view(orig_shape[0], *([1] * (len(orig_shape) - 1)))
        return dq_tensor
    else:
        max_val = tensor.abs().max()
        if max_val < 1e-8:
            return tensor
        delta = max_val / qmax
        q_tensor = torch.round(tensor / delta)
        q_tensor = torch.clamp(q_tensor, -qmax, qmax)
        dq_tensor = q_tensor * delta
        return dq_tensor

def apply_ptq_to_state_dict(state_dict, num_bits=8, per_channel=False):
    if num_bits is None:
        return state_dict
    quant_sd = {}
    for name, param in state_dict.items():
        if "weight" in name and ("conv" in name or "fc" in name or "linear" in name or "classifier" in name):
            quant_sd[name] = quantize_tensor(param.clone(), num_bits, per_channel)
        else:
            quant_sd[name] = param.clone()
    return quant_sd

# Phase Key Generators for each ablation condition
def get_phase_key_ablation(param_name, param_shape, task_idx, key_type, device):
    seed_str = f"task_{task_idx}_{param_name}"
    if key_type == 'shared':
        # Shared key uses task index 0 for all tasks, so keys are identical across tasks
        seed_str = f"task_shared_{param_name}"
        
    seed_hash = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest()[:8], 16)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_hash)
    
    if key_type in ['binary', 'shared']:
        signs = torch.randint(0, 2, size=param_shape, generator=gen, device=device) * 2 - 1
        return signs.float()
    elif key_type == 'continuous':
        # Uniform in [-1, 1]
        vals = torch.rand(size=param_shape, generator=gen, device=device) * 2.0 - 1.0
        return vals
    elif key_type == 'gaussian':
        # Gaussian N(0, 1)
        vals = torch.randn(size=param_shape, generator=gen, device=device)
        return vals
    else:
        raise ValueError(f"Unknown key type {key_type}")

# Evaluation Helper
def evaluate_model(backbone_sd, head_sd, task_name, num_bits=None, per_channel=False):
    quant_backbone = apply_ptq_to_state_dict(backbone_sd, num_bits, per_channel)
    quant_head = apply_ptq_to_state_dict(head_sd, num_bits, per_channel)
    
    backbone = get_progenitor()
    backbone.load_state_dict(quant_backbone)
    
    head = nn.Linear(512, 10)
    head.load_state_dict(quant_head)
    
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

# Calibration Helper
def calibrate_model_bn(backbone_sd, head_sd, task_name, num_bits=None, per_channel=False):
    quant_backbone = apply_ptq_to_state_dict(backbone_sd, num_bits, per_channel)
    quant_head = apply_ptq_to_state_dict(head_sd, num_bits, per_channel)
    
    backbone = get_progenitor()
    backbone.load_state_dict(quant_backbone)
    
    head = nn.Linear(512, 10)
    head.load_state_dict(quant_head)
    
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

# Run Ablation Studies
key_types = ['binary', 'continuous', 'gaussian', 'shared']
ablation_results = {}

K_sqrt = len(tasks) ** 0.5

for kt in key_types:
    print(f"\nEvaluating Ablation Configuration: Key Type = {kt.upper()}")
    ablation_results[kt] = {}
    
    # 1. Merge weights using this key type
    hsa_backbone_sd = {}
    for key in progenitor_state.keys():
        if progenitor_state[key].is_floating_point() and "weight" in key:
            hsa_update = torch.zeros_like(progenitor_state[key])
            for idx, t in enumerate(tasks):
                tau = expert_backbones[t][key] - progenitor_state[key]
                P = get_phase_key_ablation(key, progenitor_state[key].shape, idx, kt, device)
                hsa_update += tau * P
            hsa_backbone_sd[key] = progenitor_state[key] + (hsa_update / K_sqrt)
        else:
            hsa_backbone_sd[key] = progenitor_state[key].clone()
            
    # 2. Evaluate across FP32, INT8, and INT4 per-channel PTQ
    for num_bits in [None, 8, 4]:
        bit_name = 'FP32' if num_bits is None else f'INT{num_bits}'
        print(f"  Precision: {bit_name}")
        
        task_accs = {}
        for idx, t in enumerate(tasks):
            # Unbind
            retrieved_backbone_sd = {}
            for key in progenitor_state.keys():
                if progenitor_state[key].is_floating_point() and "weight" in key:
                    hsa_update_tensor = hsa_backbone_sd[key] - progenitor_state[key]
                    P = get_phase_key_ablation(key, progenitor_state[key].shape, idx, kt, device)
                    retrieved_backbone_sd[key] = progenitor_state[key] + (K_sqrt * hsa_update_tensor * P)
                else:
                    retrieved_backbone_sd[key] = hsa_backbone_sd[key].clone()
                    
            cal_backbone = calibrate_model_bn(retrieved_backbone_sd, expert_heads[t], t, num_bits, per_channel=True)
            acc = evaluate_model(cal_backbone, expert_heads[t], t, num_bits, per_channel=True)
            task_accs[t] = acc
            
        avg_acc = sum(task_accs.values()) / len(tasks)
        print(f"    Avg Accuracy: {avg_acc:.2f}% (MNIST: {task_accs['mnist']:.2f}%, FMNIST: {task_accs['fmnist']:.2f}%, CIFAR10: {task_accs['cifar10']:.2f}%)")
        ablation_results[kt][bit_name] = {
            'tasks': task_accs,
            'average': avg_acc
        }

with open("ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=4)
print("\nSaved ablation results to ablation_results.json")
