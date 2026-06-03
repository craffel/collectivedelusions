import os
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
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

# Load experts
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

def merge_models_wa(experts):
    merged = copy.deepcopy(experts[0])
    merged_state_dict = merged.state_dict()
    expert_state_dicts = [e.state_dict() for e in experts]
    
    for key in merged_state_dict.keys():
        if 'fc' in key:
            continue
        if merged_state_dict[key].dtype in [torch.float32, torch.float64, torch.bfloat16, torch.float16]:
            stacked = torch.stack([sd[key] for sd in expert_state_dicts])
            merged_state_dict[key] = torch.mean(stacked, dim=0)
    merged.load_state_dict(merged_state_dict)
    return merged

# Perform Weight Averaging
merged_model = merge_models_wa(experts)

# Map previous Conv to BN in ResNet-18
conv_to_bn = {}
prev_conv_name = None
for name, module in merged_model.named_modules():
    if isinstance(module, nn.Conv2d):
        prev_conv_name = name
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if prev_conv_name is not None:
            conv_to_bn[prev_conv_name] = name

# Perform Analytical Task-Specific BN Calibration (A-TSBC)
print("=== Evaluating Analytical Task-Specific BN Calibration (A-TSBC) ===")
accs_a_tsbc = []

for i, (loader, expert) in enumerate(zip(dataloaders, experts)):
    # Create a copy of the expert model for this task
    # This preserves the expert's task-specific running statistics (mean, variance) and affine parameters (gamma, beta)
    calib_model = copy.deepcopy(expert)
    
    # Override all non-BN weights with the merged weights (except FC)
    calib_state = calib_model.state_dict()
    merged_state = merged_model.state_dict()
    
    for key in calib_state.keys():
        if 'fc' in key:
            continue
        # Only overwrite non-BN parameters with merged weights
        # We want to keep the expert's BN running_mean, running_var, weight, and bias, but we will scale running_var
        if 'bn' not in key and 'downsample.1' not in key:
            if calib_state[key].dtype in [torch.float32, torch.float64, torch.bfloat16, torch.float16]:
                calib_state[key] = merged_state[key].clone()
                
    calib_model.load_state_dict(calib_state)
    
    # Scale the running variance of each BN layer by the Frobenius norm ratio of merged to expert Conv weights
    calib_modules = dict(calib_model.named_modules())
    expert_modules = dict(expert.named_modules())
    merged_modules = dict(merged_model.named_modules())
    
    for conv_name, bn_name in conv_to_bn.items():
        m_conv = merged_modules[conv_name]
        e_conv = expert_modules[conv_name]
        c_bn = calib_modules[bn_name]
        
        # Calculate Frobenius norm squared
        w_m = m_conv.weight.data
        w_e = e_conv.weight.data
        
        norm_m = torch.sum(w_m ** 2)
        norm_e = torch.sum(w_e ** 2)
        
        R = norm_m / (norm_e + 1e-8)
        
        # Scale the running variance of the BN layer
        c_bn.running_var.data = c_bn.running_var.data * R
        
    acc = evaluate(calib_model, loader)
    accs_a_tsbc.append(acc)

print(f"Analytical Task-Specific BN Calib (A-TSBC): MNIST: {accs_a_tsbc[0]:.4%} | FMNIST: {accs_a_tsbc[1]:.4%} | CIFAR-10: {accs_a_tsbc[2]:.4%} | Avg: {sum(accs_a_tsbc)/3:.4%}")
