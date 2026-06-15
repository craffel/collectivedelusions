import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Deep12LayerCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.ModuleList([
            ConvBlock(3, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load datasets
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
transform_color = transforms.Compose([
    transforms.ToTensor(),
])

datasets = {
    0: (torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray),
        torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)),
    1: (torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray),
        torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)),
    2: (torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_color),
        torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_color)),
    3: (torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform_color),
        torchvision.datasets.SVHN(root='./data', split='test', download=False, transform=transform_color))
}

sub_test_datasets = []
for k in range(4):
    _, test_ds = datasets[k]
    sub_test_datasets.append(Subset(test_ds, list(range(500))))

# Load checkpoints
base_model = Deep12LayerCNN().to(device)
base_model.load_state_dict(torch.load('checkpoints/base_model.pt', map_location=device))

expert_models = []
for k in range(4):
    expert = Deep12LayerCNN().to(device)
    expert.load_state_dict(torch.load(f'checkpoints/expert_{k}.pt', map_location=device))
    expert_models.append(expert)

def dare_merging_tensor(tensors, drop_rate=0.2, scaling_coef=0.3):
    stacked = torch.stack(tensors) # shape: (K, ...)
    shape = stacked.shape
    stacked_flat = stacked.view(shape[0], -1) # shape: (K, D)
    
    # DARE: Random drop and rescale
    # If drop_rate is 0, no drop
    if drop_rate > 0:
        mask = torch.rand_like(stacked_flat) >= drop_rate
        # Drop
        dropped = stacked_flat * mask
        # Rescale
        rescaled = dropped / (1.0 - drop_rate)
    else:
        rescaled = stacked_flat
        
    # Average across tasks
    merged_tv = rescaled.mean(dim=0)
    
    return merged_tv.view(shape[1:]) * scaling_coef

def evaluate_dare(drop_rate, scaling_coef):
    model = Deep12LayerCNN().to(device)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            
    # Compute and set DARE-merged weights for the backbone
    with torch.no_grad():
        for l in range(11):
            target_w = model.features[l].conv.weight
            target_b = model.features[l].conv.bias
            base_w = base_model.features[l].conv.weight
            base_b = base_model.features[l].conv.bias
            
            expert_ws = [m.features[l].conv.weight for m in expert_models]
            expert_bs = [m.features[l].conv.bias for m in expert_models]
            
            tvs_w = [ew - base_w for ew in expert_ws]
            merged_tv_w = dare_merging_tensor(tvs_w, drop_rate, scaling_coef)
            target_w.copy_(base_w + merged_tv_w)
            
            if target_b is not None:
                tvs_b = [eb - base_b for eb in expert_bs]
                merged_tv_b = dare_merging_tensor(tvs_b, drop_rate, scaling_coef)
                target_b.copy_(base_b + merged_tv_b)
                
            # BatchNorm layers
            target_bn_w = model.features[l].bn.weight
            target_bn_b = model.features[l].bn.bias
            base_bn_w = base_model.features[l].bn.weight
            base_bn_b = base_model.features[l].bn.bias
            expert_bn_ws = [m.features[l].bn.weight for m in expert_models]
            expert_bn_bs = [m.features[l].bn.bias for m in expert_models]
            
            tvs_bn_w = [ebw - base_bn_w for ebw in expert_bn_ws]
            merged_tv_bn_w = dare_merging_tensor(tvs_bn_w, drop_rate, scaling_coef)
            target_bn_w.copy_(base_bn_w + merged_tv_bn_w)
            
            if target_bn_b is not None:
                tvs_bn_b = [ebb - base_bn_b for ebb in expert_bn_bs]
                merged_tv_bn_b = dare_merging_tensor(tvs_bn_b, drop_rate, scaling_coef)
                target_bn_b.copy_(base_bn_b + merged_tv_bn_b)
                
            target_bn_rm = model.features[l].bn.running_mean
            target_bn_rv = model.features[l].bn.running_var
            base_bn_rm = base_model.features[l].bn.running_mean
            base_bn_rv = base_model.features[l].bn.running_var
            expert_bn_rms = [m.features[l].bn.running_mean for m in expert_models]
            expert_bn_rvs = [m.features[l].bn.running_var for m in expert_models]
            
            tvs_bn_rm = [ebrm - base_bn_rm for ebrm in expert_bn_rms]
            merged_tv_bn_rm = dare_merging_tensor(tvs_bn_rm, drop_rate, scaling_coef)
            target_bn_rm.copy_(base_bn_rm + merged_tv_bn_rm)
            
            tvs_bn_rv = [ebrv - base_bn_rv for ebrv in expert_bn_rvs]
            merged_tv_bn_rv = dare_merging_tensor(tvs_bn_rv, drop_rate, scaling_coef)
            target_bn_rv.copy_(base_bn_rv + merged_tv_bn_rv)

    accuracies = []
    for task_idx in range(4):
        with torch.no_grad():
            model.classifier.weight.copy_(expert_models[task_idx].classifier.weight)
            model.classifier.bias.copy_(expert_models[task_idx].classifier.bias)
        
        test_ds = sub_test_datasets[task_idx]
        loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        accuracies.append(acc)
    return accuracies

torch.manual_seed(42)
for drop_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for scaling_coef in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        accs = evaluate_dare(drop_rate, scaling_coef)
        print(f"DARE (drop={drop_rate}, coef={scaling_coef}): Accs: {accs}, Average: {np.mean(accs):.2f}%")
