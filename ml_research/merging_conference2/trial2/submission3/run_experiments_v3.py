import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Disable cuDNN to avoid initialization errors
torch.backends.cudnn.enabled = False

# Dataset directory
data_dir = "./data"

# Transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define a function to get datasets
def get_dataset(name, train=True):
    if name == "mnist":
        return torchvision.datasets.MNIST(root=data_dir, train=train, download=True, transform=transform_gray)
    elif name == "fashionmnist":
        return torchvision.datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform_gray)
    elif name == "cifar10":
        return torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform_color)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Helper to evaluate a model on a dataset with task-specific head
def evaluate(model, test_loader, expert):
    model.eval()
    # Swap in the task-specific classification head from the expert
    with torch.no_grad():
        model.fc.weight.copy_(expert.fc.weight)
        model.fc.bias.copy_(expert.fc.bias)
        
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

# Helper to load a model with specific weights
def load_model(checkpoint_path=None):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model

# Prepare datasets
datasets_list = ["mnist", "fashionmnist", "cifar10"]
test_loaders = {}

for name in datasets_list:
    full_test = get_dataset(name, train=False)
    test_loaders[name] = DataLoader(full_test, batch_size=128, shuffle=False, num_workers=2)

print("Test datasets loaded successfully.")

# Load experts
experts = {name: load_model(f"expert_{name}.pth") for name in datasets_list}

# Load pre-trained base model
base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
base_model.fc = nn.Linear(base_model.fc.in_features, 10)
base_model = base_model.to(device)

def merge_weight_averaging(experts):
    merged = load_model()
    merged_sd = merged.state_dict()
    expert_sds = [experts[name].state_dict() for name in datasets_list]
    
    for key in merged_sd.keys():
        tensors = [sd[key].float() for sd in expert_sds]
        merged_sd[key].copy_(torch.stack(tensors).mean(dim=0))
        
    merged.load_state_dict(merged_sd)
    return merged

def merge_task_arithmetic(experts, base_model, lam=0.4):
    merged = load_model()
    merged_sd = merged.state_dict()
    base_sd = base_model.state_dict()
    expert_sds = {name: experts[name].state_dict() for name in datasets_list}
    
    for key in merged_sd.keys():
        if key.startswith('fc.'):
            tensors = [expert_sds[name][key].float() for name in datasets_list]
            merged_sd[key].copy_(torch.stack(tensors).mean(dim=0))
        else:
            task_vectors = []
            for name in datasets_list:
                tv = expert_sds[name][key].float() - base_sd[key].float()
                task_vectors.append(tv)
            merged_val = base_sd[key].float() + lam * torch.stack(task_vectors).sum(dim=0)
            merged_sd[key].copy_(merged_val)
            
    merged.load_state_dict(merged_sd)
    return merged

# Native hook-free TAAC (N-TAAC)
def run_native_calibration_taac_with_size(experts, base_model, is_ta=True, lam=0.4, calibration_size=128):
    if is_ta:
        model = merge_task_arithmetic(experts, base_model, lam=lam)
    else:
        model = merge_weight_averaging(experts)
        
    # Put model in train mode to enable running statistics updates
    model.train()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Set batchnorm momentum to 1.0 (overwrite)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 1.0
            module.track_running_stats = True
            
    # Create joint calibration dataset with specific size N
    joint_images = []
    for name in datasets_list:
        full_train = get_dataset(name, train=True)
        # Always use a deterministic subset of indices for reproducibility
        indices = list(range(calibration_size))
        calib_sub = Subset(full_train, indices)
        loader = DataLoader(calib_sub, batch_size=calibration_size, shuffle=False)
        imgs, _ = next(iter(loader))
        joint_images.append(imgs)
        
    joint_images_tensor = torch.cat(joint_images, dim=0) # Shape: (K * N, 3, 32, 32)
    joint_calib_dataset = torch.utils.data.TensorDataset(joint_images_tensor, torch.zeros(len(joint_images_tensor)))
    
    # Run forward pass on the joint calibration set in a single batch
    loader = DataLoader(joint_calib_dataset, batch_size=len(joint_calib_dataset), shuffle=False)
    images, _ = next(iter(loader))
    images = images.to(device)
    
    with torch.no_grad():
        _ = model(images)
            
    # Put model back in eval mode
    model.eval()
    
    # Reset batchnorm momentum to default (0.1)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.1
            
    # Evaluate model on all tasks
    results = {}
    for name in datasets_list:
        acc = evaluate(model, test_loaders[name], experts[name])
        results[name] = acc
    results["average"] = np.mean([results[n] for n in datasets_list])
    return results

print("\n=== Running Calibration Size Sweep for N-TAAC ===")
sizes = [16, 32, 64, 128, 256, 512]

wa_sweep_results = {}
ta_sweep_results = {}

for N in sizes:
    print(f"\n--- Calibrating with N = {N} samples per task ---")
    
    # WA + N-TAAC
    res_wa = run_native_calibration_taac_with_size(experts, base_model, is_ta=False, calibration_size=N)
    wa_sweep_results[N] = res_wa
    print(f"Weight Averaging + N-TAAC (N={N}): {res_wa}")
    
    # TA (lambda=0.4) + N-TAAC
    res_ta = run_native_calibration_taac_with_size(experts, base_model, is_ta=True, lam=0.4, calibration_size=N)
    ta_sweep_results[N] = res_ta
    print(f"Task Arithmetic (lam=0.4) + N-TAAC (N={N}): {res_ta}")

print("\n=== FINAL RESULTS SUMMARY ===")
print("N\tWA Average Acc (%)\tTA (lam=0.4) Average Acc (%)")
for N in sizes:
    print(f"{N}\t{wa_sweep_results[N]['average']:.2f}%\t\t\t{ta_sweep_results[N]['average']:.2f}%")

print("\nDetails:")
print("WA Details:")
for N in sizes:
    print(f"N={N}: MNIST={wa_sweep_results[N]['mnist']:.2f}%, F-MNIST={wa_sweep_results[N]['fashionmnist']:.2f}%, CIFAR-10={wa_sweep_results[N]['cifar10']:.2f}%, Avg={wa_sweep_results[N]['average']:.2f}%")
print("TA Details:")
for N in sizes:
    print(f"N={N}: MNIST={ta_sweep_results[N]['mnist']:.2f}%, F-MNIST={ta_sweep_results[N]['fashionmnist']:.2f}%, CIFAR-10={ta_sweep_results[N]['cifar10']:.2f}%, Avg={ta_sweep_results[N]['average']:.2f}%")

print("\nCalibration size sweep finished successfully.")
