import os
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Subset, DataLoader

# Load datasets in the same way
transform_mnist = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])
transform_fmnist = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
])
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fmnist)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

test_loader_mnist = DataLoader(test_mnist, batch_size=256, shuffle=False)
test_loader_fmnist = DataLoader(test_fmnist, batch_size=256, shuffle=False)
test_loader_cifar = DataLoader(test_cifar, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load models
progenitor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

expert_mnist = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
expert_mnist.fc = nn.Linear(512, 10)
expert_mnist.load_state_dict(torch.load("expert_mnist_severe.pth", map_location=device))

expert_fmnist = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
expert_fmnist.fc = nn.Linear(512, 10)
expert_fmnist.load_state_dict(torch.load("expert_fmnist_severe.pth", map_location=device))

expert_cifar = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
expert_cifar.fc = nn.Linear(512, 10)
expert_cifar.load_state_dict(torch.load("expert_cifar_severe.pth", map_location=device))

experts = [expert_mnist, expert_fmnist, expert_cifar]
dataloaders = [test_loader_mnist, test_loader_fmnist, test_loader_cifar]

def evaluate(model, dataloader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return correct / total

def merge_models(progenitor, experts, method='WA', lam=0.5):
    merged = copy.deepcopy(progenitor)
    merged_state_dict = merged.state_dict()
    expert_state_dicts = [e.state_dict() for e in experts]
    
    for key in merged_state_dict.keys():
        if 'fc' in key:
            continue
        if merged_state_dict[key].dtype in [torch.float32, torch.float64, torch.bfloat16, torch.float16]:
            if 'running_mean' in key or 'running_var' in key:
                # In standard merge, we average. In task-specific, we'll swap them during evaluation.
                stacked = torch.stack([sd[key] for sd in expert_state_dicts])
                merged_state_dict[key] = torch.mean(stacked, dim=0)
            else:
                if method == 'WA':
                    stacked = torch.stack([sd[key] for sd in expert_state_dicts])
                    merged_state_dict[key] = torch.mean(stacked, dim=0)
                elif method == 'TA':
                    prog_val = progenitor.state_dict()[key]
                    updates = [sd[key] - prog_val for sd in expert_state_dicts]
                    merged_state_dict[key] = prog_val + lam * sum(updates)
    merged.load_state_dict(merged_state_dict)
    return merged

def get_fnbc_ratios(merged_model, experts):
    conv_to_bn = {}
    prev_conv_name = None
    for name, module in merged_model.named_modules():
        if isinstance(module, nn.Conv2d):
            prev_conv_name = name
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if prev_conv_name is not None:
                conv_to_bn[prev_conv_name] = name
                
    merged_modules = dict(merged_model.named_modules())
    expert_modules_lists = [dict(e.named_modules()) for e in experts]
    
    ratios = {}
    for conv_name, bn_name in conv_to_bn.items():
        merged_conv = merged_modules[conv_name]
        expert_convs = [em[conv_name] for em in expert_modules_lists]
        
        w_merged = merged_conv.weight.data
        w_experts = [ec.weight.data for ec in expert_convs]
        
        norm_merged = torch.sum(w_merged ** 2)
        norm_experts = torch.mean(torch.stack([torch.sum(we.to(w_merged.device) ** 2) for we in w_experts]))
        
        R_l = norm_merged / (norm_experts + 1e-8)
        R_l = torch.clamp(R_l, min=0.1, max=10.0)
        ratios[bn_name] = R_l
    return ratios

# Let's evaluate WA
print("=== Evaluating WA ===")
m_model = merge_models(progenitor, experts, method='WA')
ratios = get_fnbc_ratios(m_model, experts)

# 1. Standard WA (No Calib)
accs_standard = []
for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
    m_model.fc = exp.fc
    # Keep the averaged BN stats
    acc = evaluate(m_model, loader)
    accs_standard.append(acc)
print(f"Standard WA: MNIST: {accs_standard[0]:.4%} | FMNIST: {accs_standard[1]:.4%} | CIFAR-10: {accs_standard[2]:.4%} | Avg: {sum(accs_standard)/3:.4%}")

# 2. WA with FNBC (Standard)
# Let's apply FNBC (in-place on m_model)
m_model_fnbc = copy.deepcopy(m_model)
for bn_name, R_l in ratios.items():
    bn_module = dict(m_model_fnbc.named_modules())[bn_name]
    bn_module.running_var.copy_(R_l.to(bn_module.running_var.device) * bn_module.running_var)
accs_fnbc = []
for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
    m_model_fnbc.fc = exp.fc
    acc = evaluate(m_model_fnbc, loader)
    accs_fnbc.append(acc)
print(f"Standard FNBC: MNIST: {accs_fnbc[0]:.4%} | FMNIST: {accs_fnbc[1]:.4%} | CIFAR-10: {accs_fnbc[2]:.4%} | Avg: {sum(accs_fnbc)/3:.4%}")

# 3. WA with Task-Specific BN Statistics (No scaling)
# For each task, we swap the BN running stats with the expert's original running stats!
accs_ts = []
for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
    m_model_ts = copy.deepcopy(m_model)
    m_model_ts.fc = exp.fc
    
    # Copy expert BN stats to the merged model
    expert_modules = dict(exp.named_modules())
    merged_modules = dict(m_model_ts.named_modules())
    for name, module in m_model_ts.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            exp_bn = expert_modules[name]
            module.running_mean.copy_(exp_bn.running_mean)
            module.running_var.copy_(exp_bn.running_var)
            
    acc = evaluate(m_model_ts, loader)
    accs_ts.append(acc)
print(f"Task-Specific BN (No Scaling): MNIST: {accs_ts[0]:.4%} | FMNIST: {accs_ts[1]:.4%} | CIFAR-10: {accs_ts[2]:.4%} | Avg: {sum(accs_ts)/3:.4%}")

# 4. WA with Task-Specific FNBC (TS-FNBC)
# Swap BN stats AND scale them by R_l!
accs_ts_fnbc = []
for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
    m_model_ts_fnbc = copy.deepcopy(m_model)
    m_model_ts_fnbc.fc = exp.fc
    
    # Copy expert BN stats and apply R_l scaling
    expert_modules = dict(exp.named_modules())
    merged_modules = dict(m_model_ts_fnbc.named_modules())
    for name, module in m_model_ts_fnbc.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            exp_bn = expert_modules[name]
            module.running_mean.copy_(exp_bn.running_mean)
            
            R_l = ratios[name]
            scaled_var = R_l.to(exp_bn.running_var.device) * exp_bn.running_var
            module.running_var.copy_(scaled_var)
            
    acc = evaluate(m_model_ts_fnbc, loader)
    accs_ts_fnbc.append(acc)
print(f"Task-Specific FNBC (TS-FNBC): MNIST: {accs_ts_fnbc[0]:.4%} | FMNIST: {accs_ts_fnbc[1]:.4%} | CIFAR-10: {accs_ts_fnbc[2]:.4%} | Avg: {sum(accs_ts_fnbc)/3:.4%}")
