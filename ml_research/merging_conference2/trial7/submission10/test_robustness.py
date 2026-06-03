import os
import copy
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Subset, DataLoader

class BlendedBatchNorm2d(nn.Module):
    def __init__(self, original_bn, alpha=1.0):
        super().__init__()
        self.original_bn = original_bn
        self.alpha = alpha
        
    def forward(self, x):
        if self.alpha > 0:
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            merged_mean = self.original_bn.running_mean
            merged_var = self.original_bn.running_var
            
            active_mean = (1 - self.alpha) * merged_mean + self.alpha * batch_mean
            active_var = (1 - self.alpha) * merged_var + self.alpha * batch_var
        else:
            active_mean = self.original_bn.running_mean
            active_var = self.original_bn.running_var
            
        w = self.original_bn.weight.view(1, -1, 1, 1)
        b = self.original_bn.bias.view(1, -1, 1, 1)
        eps = self.original_bn.eps
        
        x_norm = (x - active_mean.view(1, -1, 1, 1)) / torch.sqrt(active_var.view(1, -1, 1, 1) + eps)
        return w * x_norm + b

def apply_spttbc(model, alpha=1.0):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]
            setattr(parent, attr_name, BlendedBatchNorm2d(module, alpha=alpha))

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

def evaluate(model, dataloader, noise_std=0.0):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if noise_std > 0.0:
                x = x + torch.randn_like(x) * noise_std
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

def calibrate_for_task(model, task_loader, num_batches=20):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
            
    with torch.no_grad():
        count = 0
        for x, _ in task_loader:
            x = x.to(device)
            _ = model(x)
            count += 1
            if count >= num_batches:
                break
    return model

calib_loaders = [
    DataLoader(train_mnist_sub, batch_size=64, shuffle=True),
    DataLoader(train_fmnist_sub, batch_size=64, shuffle=True),
    DataLoader(train_cifar_sub, batch_size=64, shuffle=True)
]

print("=== Evaluating WA under Test-Time Noise ===")
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
results = {}

for std in noise_levels:
    print(f"\n--- Noise Std = {std} ---")
    results[std] = {}
    
    # 1. No Calibration
    m_no_calib = merge_models(progenitor, experts, method='WA')
    accs_no_calib = []
    for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
        m_no_calib.fc = exp.fc
        accs_no_calib.append(evaluate(m_no_calib, loader, noise_std=std))
    avg_no_calib = sum(accs_no_calib) / 3
    results[std]["No_Calib"] = avg_no_calib
    print(f"No Calib   - Avg Acc: {avg_no_calib:.4%}")
    
    # 2. SP-TTBC (Test-Time active adaptation, batch=64)
    # Using small batch size 64 for SP-TTBC test
    test_loader_mnist_64 = DataLoader(test_mnist, batch_size=64, shuffle=False)
    test_loader_fmnist_64 = DataLoader(test_fmnist, batch_size=64, shuffle=False)
    test_loader_cifar_64 = DataLoader(test_cifar, batch_size=64, shuffle=False)
    dataloaders_64 = [test_loader_mnist_64, test_loader_fmnist_64, test_loader_cifar_64]
    
    m_spttbc = merge_models(progenitor, experts, method='WA')
    apply_spttbc(m_spttbc, alpha=1.0)
    accs_spttbc = []
    for i, (loader, exp) in enumerate(zip(dataloaders_64, experts)):
        m_spttbc.fc = exp.fc
        accs_spttbc.append(evaluate(m_spttbc, loader, noise_std=std))
    avg_spttbc = sum(accs_spttbc) / 3
    results[std]["SP-TTBC"] = avg_spttbc
    print(f"SP-TTBC    - Avg Acc: {avg_spttbc:.4%}")
    
    # 3. TS-BC (Offline task-specific)
    accs_ts_bc = []
    for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
        m_tsbc = merge_models(progenitor, experts, method='WA')
        calibrate_for_task(m_tsbc, calib_loaders[i], num_batches=20)
        m_tsbc.fc = exp.fc
        accs_ts_bc.append(evaluate(m_tsbc, loader, noise_std=std))
    avg_ts_bc = sum(accs_ts_bc) / 3
    results[std]["TS-BC"] = avg_ts_bc
    print(f"TS-BC      - Avg Acc: {avg_ts_bc:.4%}")

# Save results to json
with open("robustness_results.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=4)
print("\nRobustness results saved to robustness_results.json!")
