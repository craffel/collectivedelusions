import torch
import torch.nn as nn
from models import ResNet18Backbone, MLPBackbone, CompleteModel
import merging
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def get_loaders(task_name):
    # Grayscale transforms (Resize to 32x32, duplicate channels, normalize)
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # CIFAR-10 transforms (Resize to 32x32, normalize)
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

# Merging methods
merges = {
    "WA": merging.weight_averaging(experts),
    "TIES": merging.ties_merging(experts, progenitor, fraction=0.2),
    "WCPR": merging.wcpr_merging(experts, progenitor),
    "QR-SP-WCPR": merging.qr_sp_wcpr_merging(experts, progenitor, sign_merger='ties', fraction=0.2, gamma=2.0, scale_compensation=True)
}

for name, state in merges.items():
    print(f"\n--- Method: {name} ---")
    for use_de_bn in [False, True]:
        bb = ResNet18Backbone().to(device)
        bb.load_state_dict(state)
        if use_de_bn:
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
            score = 100.0 * correct / total
            scores.append(score)
            print(f"  Task {task} (DE-BN: {use_de_bn}): {score:.2f}%")
        print(f"  Average Accuracy (DE-BN: {use_de_bn}): {np.mean(scores):.2f}%")
