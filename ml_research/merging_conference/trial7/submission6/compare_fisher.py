import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class ResNet18Custom(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.base_model.fc(feat)
        return feat, logits

def get_resnet18_1channel():
    model = resnet18()
    conv1_new = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = conv1_new
    model.fc = nn.Linear(512, 10)
    return model

def compute_batch_fisher_fast(expert_model, batch_X, device):
    fisher = {name: torch.zeros_like(p) for name, p in expert_model.named_parameters() if p.requires_grad}
    expert_model.eval()
    
    _, logits = expert_model(batch_X)
    pseudo_labels = torch.argmax(logits, dim=-1)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, pseudo_labels)
    expert_model.zero_grad()
    loss.backward()
    
    for name, param in expert_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher[name] = param.grad.data ** 2
            
    return fisher

def get_joint_fisher_fast(expert_models, batch_X, device):
    joint_fisher = {}
    K = len(expert_models)
    for k in range(K):
        fisher_k = compute_batch_fisher_fast(expert_models[k], batch_X, device)
        for name, val in fisher_k.items():
            clean_name = name.replace('base_model.', '')
            tensor_avg = val.mean().item()
            if clean_name not in joint_fisher:
                joint_fisher[clean_name] = 0.0
            joint_fisher[clean_name] += tensor_avg / K
    return joint_fisher

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    mnist_loader = DataLoader(Subset(mnist_test, list(range(100))), batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(Subset(fmnist_test, list(range(100))), batch_size=64, shuffle=False)

    mnist_batch = next(iter(mnist_loader))[0].to(device)
    fmnist_batch = next(iter(fmnist_loader))[0].to(device)

    # Load experts
    expert_paths = {
        'mnist': 'mnist_expert.pt',
        'kmnist': 'kmnist_expert.pt',
        'fashionmnist': 'fashionmnist_expert.pt'
    }
    experts = {}
    for name, path in expert_paths.items():
        model = get_resnet18_1channel()
        model.load_state_dict(torch.load(path, map_location=device))
        model = ResNet18Custom(model).to(device)
        model.eval()
        experts[name] = model

    expert_list = [experts['mnist'], experts['kmnist'], experts['fashionmnist']]

    fisher_mnist = get_joint_fisher_fast(expert_list, mnist_batch, device)
    fisher_fmnist = get_joint_fisher_fast(expert_list, fmnist_batch, device)

    print(f"{'Layer Name':<40} | {'MNIST Fisher':<12} | {'F-MNIST Fisher':<12} | {'Ratio':<8}")
    print("-" * 80)
    for name in fisher_mnist.keys():
        v1 = fisher_mnist[name]
        v2 = fisher_fmnist[name]
        ratio = v2 / (v1 + 1e-12)
        print(f"{name:<40} | {v1:<12.6f} | {v2:<12.6f} | {ratio:<8.4f}")

if __name__ == '__main__':
    main()
