import torch
import torch.nn as nn
from models import ResNet18Backbone, MLPBackbone, CompleteModel
import merging
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def get_loaders(task_name):
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    if task_name == 'mnist':
        ds = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)
        train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray)
    elif task_name == 'fmnist':
        ds = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)
        train_ds = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray)
    elif task_name == 'cifar10':
        ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_cifar)
        train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_cifar)
    subset = Subset(ds, range(200))
    return {
        'test': DataLoader(subset, batch_size=64, shuffle=False),
        'train': DataLoader(train_ds, batch_size=64, shuffle=True)
    }

def apply_de_bn(model, loaders, N=16):
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None
    with torch.no_grad():
        for task_name in ['mnist', 'fmnist', 'cifar10']:
            train_loader = loaders[task_name]['train']
            samples_drawn = 0
            for x, _ in train_loader:
                x = x.to(device)
                batch_size = x.size(0)
                if samples_drawn + batch_size > N:
                    x = x[:N - samples_drawn]
                model(x)
                samples_drawn += x.size(0)
                if samples_drawn >= N:
                    break
    model.eval()
    for p in model.parameters():
        p.requires_grad = True

def evaluate_merge_state(state, loaders, heads):
    bb = ResNet18Backbone().to(device)
    bb.load_state_dict(state)
    apply_de_bn(bb, loaders, N=16)
    
    scores = []
    for task in ['mnist', 'fmnist', 'cifar10']:
        model = CompleteModel(bb, heads[task])
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loaders[task]['test']:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, pred = outputs.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        scores.append(100.0 * correct / total)
    return np.mean(scores)

# Load progenitor and experts
progenitor = ResNet18Backbone().to(device)
progenitor.load_state_dict(torch.load('checkpoints/resnet18_progenitor.pt', map_location=device))

experts = []
heads = {}
for task in ['mnist', 'fmnist', 'cifar10']:
    exp = ResNet18Backbone().to(device)
    exp.load_state_dict(torch.load(f'checkpoints/resnet18_{task}_backbone.pt', map_location=device))
    experts.append(exp)
    
    head = nn.Linear(512, 10).to(device)
    head.load_state_dict(torch.load(f'checkpoints/resnet18_{task}_head.pt', map_location=device))
    heads[task] = head

loaders = {task: get_loaders(task) for task in ['mnist', 'fmnist', 'cifar10']}

# Let's test different OT merging formulations on sparse weights
# TIES baseline
base_ties = merging.ties_merging(experts, progenitor, fraction=0.2)
ties_avg = evaluate_merge_state(base_ties, loaders, heads)
print(f"Baseline TIES-Merging average accuracy: {ties_avg:.2f}%")

# Let's define several variants of SP-WCPR and evaluate them:
# Variant 1: Original implementation of QR-SP-WCPR (dense OT then mask)
print("\nEvaluating Variant 1: Dense OT then Mask (current implementation)")
v1_state = merging.qr_sp_wcpr_merging(experts, progenitor, sign_merger='ties', fraction=0.2, gamma=2.0, scale_compensation=True)
v1_avg = evaluate_merge_state(v1_state, loaders, heads)
print(f"Variant 1 Average Accuracy: {v1_avg:.2f}%")

# Variant 2: OT performed ONLY on the active elements of each channel
# To do this, for each channel, we extract the non-zero elements of the sign-resolved (TIES) update.
# Let's write the OT logic for active elements:
def active_ot_merging(backbones, initial_backbone, fraction=0.2, gamma=2.0, scale_compensation=True):
    merged = {}
    state_init = initial_backbone.state_dict()
    task_vectors = merging.get_task_vectors(backbones, initial_backbone)
    
    # Compute base TIES-Merged update
    base_ties = merging.ties_merging(backbones, initial_backbone, fraction=fraction)
    
    for key in state_init.keys():
        tensor_init = state_init[key]
        if not tensor_init.dtype.is_floating_point:
            merged[key] = tensor_init.clone()
            continue
            
        tvs = [tv[key] for tv in task_vectors]
        # Resolved update
        ties_update = base_ties[key] - tensor_init
        
        if tensor_init.dim() >= 2:
            C_out = tensor_init.size(0)
            T_calibrated = torch.zeros_like(ties_update)
            
            for c in range(C_out):
                # Mask for active elements in channel c
                mask_c = (ties_update[c] != 0)
                active_indices = torch.nonzero(mask_c, as_tuple=True)
                
                m_c_active = ties_update[c][active_indices]
                if m_c_active.numel() == 0:
                    continue
                    
                # Sort indices of active merged elements
                I_c = torch.argsort(m_c_active)
                
                # Get the sorted active elements for experts at the same coordinates
                s_ks = []
                for tv in tvs:
                    # Extract elements from the same active coordinates
                    tv_c_active = tv[c][active_indices]
                    s_ks.append(torch.sort(tv_c_active)[0])
                    
                s_target_c = torch.mean(torch.stack(s_ks), dim=0)
                
                # Optional standard-deviation based outlier clamping
                median_c = torch.median(s_target_c)
                std_c = torch.std(s_target_c)
                s_target_c_clamped = torch.clamp(s_target_c, min=median_c - gamma * std_c, max=median_c + gamma * std_c)
                
                # Assign back to sorted active coordinates
                c_flat = torch.zeros_like(m_c_active)
                c_flat[I_c] = s_target_c_clamped
                
                # Write back to calibrated update
                T_calibrated[c][active_indices] = c_flat
                
            if scale_compensation:
                p_c = (ties_update != 0).float().mean().item()
                T_calibrated = T_calibrated / math.sqrt(max(p_c, 1e-5))
                
            merged[key] = tensor_init + T_calibrated
        else:
            merged[key] = base_ties[key].clone()
            
    return merged

print("\nEvaluating Variant 2: Active-Only OT Mapping")
v2_state = active_ot_merging(experts, progenitor, fraction=0.2, gamma=2.0, scale_compensation=True)
v2_avg = evaluate_merge_state(v2_state, loaders, heads)
print(f"Variant 2 Average Accuracy: {v2_avg:.2f}%")

# Variant 3: What if we don't do scale compensation in Variant 2?
print("\nEvaluating Variant 3: Active-Only OT Mapping (No Scale Compensation)")
v3_state = active_ot_merging(experts, progenitor, fraction=0.2, gamma=2.0, scale_compensation=False)
v3_avg = evaluate_merge_state(v3_state, loaders, heads)
print(f"Variant 3 Average Accuracy: {v3_avg:.2f}%")
