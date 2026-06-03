import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np

# Use CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Normalization transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloader(task_name, is_train=True, subset_size=128, seed=42):
    np.random.seed(seed)
    if task_name == "mnist":
        dataset = torchvision.datasets.MNIST(root='./data', train=is_train, download=False, transform=transform_gray)
    elif task_name == "fashion":
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=is_train, download=False, transform=transform_gray)
    else:
        raise ValueError()
        
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    subset_dataset = Subset(dataset, indices)
    # Shuffle loader differently based on seed
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(subset_dataset, batch_size=32, shuffle=True, generator=generator)
    return loader

def get_full_test_loader(task_name):
    if task_name == "mnist":
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)
    elif task_name == "fashion":
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)
    else:
        raise ValueError()
    indices = np.random.choice(len(dataset), 500, replace=False)
    subset_dataset = Subset(dataset, indices)
    loader = DataLoader(subset_dataset, batch_size=128, shuffle=False)
    return loader

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return 100.0 * correct / total

def calibrate_bn_momentum(model, loader, momentum=0.1):
    model.train()
    # Reset stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = momentum
            
    # Forward pass over the calibration set
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)
    model.eval()

def calibrate_bn_exact(model, loader):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
            
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)
    model.eval()

# Load experts
experts = {}
for task in ["mnist", "fashion"]:
    ckpt_path = f"checkpoints/{task}_expert.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model_state = model.state_dict()
    model_state.update(checkpoint['backbone_state_dict'])
    for k, v in checkpoint['fc_state_dict'].items():
        model_state[f"fc.{k}"] = v
    model.load_state_dict(model_state)
    experts[task] = model.to(device)

# Merge backbones (Weight Averaging of MNIST and Fashion experts)
merged_model = resnet18().to(device)
merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)

def do_wa():
    merged_state = merged_model.state_dict()
    e_states = [experts["mnist"].state_dict(), experts["fashion"].state_dict()]
    for k in merged_state.keys():
        if merged_state[k].is_floating_point():
            merged_state[k] = torch.stack([state[k] for state in e_states]).mean(dim=0)
        else:
            merged_state[k] = e_states[0][k].clone()
    merged_model.load_state_dict(merged_state)

# Test loaders
test_loaders = {task: get_full_test_loader(task) for task in ["mnist", "fashion"]}

seeds = [42, 100, 2026, 999, 12345]
momentum_results = []
exact_results = []

print("\n--- Running Momentum-Based BN Calibration Sweep ---")
for seed in seeds:
    do_wa()
    # Loader for this seed
    mnist_loader = get_dataloader("mnist", is_train=True, subset_size=128, seed=seed)
    calibrate_bn_momentum(merged_model, mnist_loader, momentum=0.1)
    acc = evaluate(merged_model, test_loaders["mnist"])
    momentum_results.append(acc)
    print(f"Seed {seed}: MNIST Accuracy = {acc:.2f}%")

print("\n--- Running Exact Cumulative BN Calibration Sweep ---")
for seed in seeds:
    do_wa()
    mnist_loader = get_dataloader("mnist", is_train=True, subset_size=128, seed=seed)
    calibrate_bn_exact(merged_model, mnist_loader)
    acc = evaluate(merged_model, test_loaders["mnist"])
    exact_results.append(acc)
    print(f"Seed {seed}: MNIST Accuracy = {acc:.2f}%")

print("\n=============================================")
print("STATISTICAL SUMMARY (MNIST WA CALIBRATION)")
print("=============================================")
print(f"Momentum (0.1): Mean = {np.mean(momentum_results):.2f}%, Std = {np.std(momentum_results):.2f}%")
print(f"Exact (None):   Mean = {np.mean(exact_results):.2f}%, Std = {np.std(exact_results):.2f}%")
print("=============================================")
