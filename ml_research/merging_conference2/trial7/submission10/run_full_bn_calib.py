import os
import copy
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Subset, DataLoader

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

train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fmnist)
train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)

def get_subset(dataset, num_samples=5000, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:num_samples].tolist()
    return Subset(dataset, indices)

train_mnist_sub = get_subset(train_mnist)
train_fmnist_sub = get_subset(train_fmnist)
train_cifar_sub = get_subset(train_cifar)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def standard_bn_calibration(model, calibration_loaders, num_batches=20):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
            
    with torch.no_grad():
        for loader in calibration_loaders:
            count = 0
            for x, _ in loader:
                x = x.to(device)
                _ = model(x)
                count += 1
                if count >= num_batches:
                    break

calib_loaders = [
    DataLoader(train_mnist_sub, batch_size=64, shuffle=True),
    DataLoader(train_fmnist_sub, batch_size=64, shuffle=True),
    DataLoader(train_cifar_sub, batch_size=64, shuffle=True)
]

lambdas = [0.1, 0.3, 0.5, 0.7, 1.0]
merge_configs = [('WA', None)] + [('TA', l) for l in lambdas]

results = {}
for method, lam in merge_configs:
    config_name = f"{method} (lambda={lam})" if lam is not None else "WA"
    m_model = merge_models(progenitor, experts, method=method, lam=lam if lam is not None else 0.5)
    
    # Run Standard BN Calibration
    m_model_calib = copy.deepcopy(m_model)
    m_model_calib.fc = nn.Linear(512, 10)
    standard_bn_calibration(m_model_calib, calib_loaders, num_batches=20)
    
    accs = []
    for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
        m_model_calib.fc = exp.fc
        acc = evaluate(m_model_calib, loader)
        accs.append(acc)
    avg_acc = sum(accs) / 3
    print(f"{config_name:<20} | MNIST: {accs[0]:.4%} | FMNIST: {accs[1]:.4%} | CIFAR-10: {accs[2]:.4%} | Avg: {avg_acc:.4%}")
    results[config_name] = accs + [avg_acc]
