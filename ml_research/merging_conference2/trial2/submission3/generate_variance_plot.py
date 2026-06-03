import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt

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
calibration_size = 128
calib_datasets = {}
test_loaders = {}

for name in datasets_list:
    full_train = get_dataset(name, train=True)
    indices = list(range(calibration_size))
    calib_datasets[name] = Subset(full_train, indices)
    
    full_test = get_dataset(name, train=False)
    test_loaders[name] = DataLoader(full_test, batch_size=128, shuffle=False, num_workers=2)

print("Datasets loaded.")

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

# Get uncalibrated WA baseline
wa_model = merge_weight_averaging(experts)

# Get WA + TCAC (calibrated for CIFAR-10)
def calibrate_tcac_for_cifar10(experts, base_model):
    # Get expert statistics
    stats_orig = {}
    experts["cifar10"].eval()
    expert_hooks = []
    
    def make_expert_hook(bn_name):
        def hook(module, inp, out):
            x = inp[0].detach()
            mean = x.mean(dim=(0, 2, 3))
            std = torch.sqrt(x.var(dim=(0, 2, 3), unbiased=False) + 1e-5)
            stats_orig[bn_name] = (mean, std)
        return hook
        
    for bn_name, module in experts["cifar10"].named_modules():
        if isinstance(module, nn.BatchNorm2d):
            expert_hooks.append(module.register_forward_hook(make_expert_hook(bn_name)))
            
    loader = DataLoader(calib_datasets["cifar10"], batch_size=calibration_size, shuffle=False)
    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        _ = experts["cifar10"](images)
    for h in expert_hooks:
        h.remove()
        
    # Get merged model
    model = merge_weight_averaging(experts)
    model.eval()
    
    stats_merged = {}
    merged_hooks = []
    
    def make_merged_hook(bn_name):
        def hook(module, inp):
            x = inp[0]
            mean = x.mean(dim=(0, 2, 3))
            std = torch.sqrt(x.var(dim=(0, 2, 3), unbiased=False) + 1e-5)
            stats_merged[bn_name] = (mean, std)
            
            mean_orig, std_orig = stats_orig[bn_name]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
            mean_orig = mean_orig.view(1, -1, 1, 1)
            std_orig = std_orig.view(1, -1, 1, 1)
            
            x_calib = ((x - mean) / std) * std_orig + mean_orig
            return (x_calib,)
        return hook
        
    for bn_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            merged_hooks.append(module.register_forward_pre_hook(make_merged_hook(bn_name)))
            
    loader = DataLoader(calib_datasets["cifar10"], batch_size=calibration_size, shuffle=False)
    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        _ = model(images)
    for h in merged_hooks:
        h.remove()
        
    # Set the running stats
    expert_state_dict = experts["cifar10"].state_dict()
    with torch.no_grad():
        for bn_name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                mean, std = stats_merged[bn_name]
                module.running_mean.copy_(mean)
                module.running_var.copy_(torch.clamp(std**2 - 1e-5, min=1e-5))
                module.weight.copy_(expert_state_dict[f"{bn_name}.weight"])
                module.bias.copy_(expert_state_dict[f"{bn_name}.bias"])
                
    return model

tcac_model = calibrate_tcac_for_cifar10(experts, base_model)

# Get WA + N-TAAC (Ours)
def calibrate_ntaac(experts):
    model = merge_weight_averaging(experts)
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 1.0
            module.track_running_stats = True
            
    # Create joint calibration set
    joint_images = []
    for name in datasets_list:
        loader = DataLoader(calib_datasets[name], batch_size=calibration_size, shuffle=False)
        imgs, _ = next(iter(loader))
        joint_images.append(imgs)
    joint_images_tensor = torch.cat(joint_images, dim=0) # Shape: 384, 3, 32, 32
    
    images = joint_images_tensor.to(device)
    with torch.no_grad():
        _ = model(images)
        
    model.eval()
    return model

ntaac_model = calibrate_ntaac(experts)

# Now, we define a function to measure layer-wise input activation variances
def measure_variances(model, test_loader):
    model.eval()
    variances = []
    hooks = []
    
    def hook_fn(module, inp):
        x = inp[0].detach()
        # Compute channel-wise variance and average across channels
        channel_vars = x.var(dim=(0, 2, 3), unbiased=False)
        mean_var = channel_vars.mean().item()
        variances.append(mean_var)
        
    # Find all BatchNorm layers and register forward pre-hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_pre_hook(hook_fn))
            
    # Get a batch of test data from CIFAR-10
    batch_size = 128
    loader = DataLoader(get_dataset("cifar10", train=False), batch_size=batch_size, shuffle=False)
    images, _ = next(iter(loader))
    images = images.to(device)
    
    with torch.no_grad():
        _ = model(images)
        
    for h in hooks:
        h.remove()
        
    return variances

# Measure for all 4 configurations
cifar10_expert = experts["cifar10"]
loader = test_loaders["cifar10"]

print("Measuring variances...")
vars_expert = measure_variances(cifar10_expert, loader)
vars_wa = measure_variances(wa_model, loader)
vars_tcac = measure_variances(tcac_model, loader)
vars_ntaac = measure_variances(ntaac_model, loader)

print("Expert Variances:", [round(v, 4) for v in vars_expert])
print("WA Variances:", [round(v, 4) for v in vars_wa])
print("TCAC Variances:", [round(v, 4) for v in vars_tcac])
print("N-TAAC Variances:", [round(v, 4) for v in vars_ntaac])

# Let's verify that the layer indices match
layers = list(range(1, len(vars_expert) + 1))

# Plot the curves beautifully
plt.figure(figsize=(8, 4.5))

plt.plot(layers, vars_expert, marker='o', linestyle='-', color='#2ca02c', linewidth=2, label='CIFAR-10 Expert (Upper Bound)')
plt.plot(layers, vars_wa, marker='x', linestyle='--', color='#d62728', linewidth=2, label='WA Baseline (Uncalibrated)')
plt.plot(layers, vars_tcac, marker='s', linestyle=':', color='#ff7f0e', linewidth=2, label='WA + TCAC (Task-Conditional)')
plt.plot(layers, vars_ntaac, marker='^', linestyle='-.', color='#1f77b4', linewidth=2, label='WA + N-TAAC (Ours, Task-Agnostic)')

plt.yscale('log')
plt.xlabel('BatchNorm Layer Index (Depth)', fontsize=12)
plt.ylabel('Average Activation Variance (Log Scale)', fontsize=12)
plt.title('Empirical Deconstruction of Variance Collapse and Restoration', fontsize=13, fontweight='bold')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xticks(layers)
plt.legend(fontsize=11, loc='lower left')
plt.tight_layout()

# Save the plot
plt.savefig('variance_collapse.pdf')
plt.savefig('variance_collapse.png')
print("Variance plot saved as variance_collapse.pdf and variance_collapse.png")
