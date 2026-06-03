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

# Load datasets (N = 128 per task for calibration)
calibration_size = 128
calib_datasets = {}
test_loaders = {}

datasets_list = ["mnist", "fashionmnist", "cifar10"]

for name in datasets_list:
    full_train = get_dataset(name, train=True)
    indices = list(range(calibration_size))
    calib_datasets[name] = Subset(full_train, indices)
    
    # Evaluate on full test sets
    full_test = get_dataset(name, train=False)
    test_loaders[name] = DataLoader(full_test, batch_size=128, shuffle=False, num_workers=2)

print("Datasets loaded successfully.")

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

# Evaluate a model on all tasks (with no calibration)
def evaluate_multi_task(model):
    results = {}
    for name in datasets_list:
        acc = evaluate(model, test_loaders[name], experts[name])
        results[name] = acc
    results["average"] = np.mean([results[n] for n in datasets_list])
    return results

# --- Original Hook-based TCAC ---

def run_sequential_calibration_tcac(experts, base_model, is_ta=True, lam=0.4):
    stats_orig = {}
    for name in datasets_list:
        stats_orig[name] = {}
        experts[name].eval()
        expert_hooks = []
        def make_expert_hook(bn_name):
            def hook(module, inp, out):
                x = inp[0].detach()
                mean = x.mean(dim=(0, 2, 3))
                std = torch.sqrt(x.var(dim=(0, 2, 3), unbiased=False) + 1e-5)
                stats_orig[name][bn_name] = (mean, std)
            return hook
            
        for bn_name, module in experts[name].named_modules():
            if isinstance(module, nn.BatchNorm2d):
                expert_hooks.append(module.register_forward_hook(make_expert_hook(bn_name)))
                
        loader = DataLoader(calib_datasets[name], batch_size=calibration_size, shuffle=False)
        images, _ = next(iter(loader))
        images = images.to(device)
        with torch.no_grad():
            _ = experts[name](images)
        for h in expert_hooks:
            h.remove()

    results = {}
    for name in datasets_list:
        if is_ta:
            model = merge_task_arithmetic(experts, base_model, lam=lam)
        else:
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
                
                mean_orig, std_orig = stats_orig[name][bn_name]
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
                
        loader = DataLoader(calib_datasets[name], batch_size=calibration_size, shuffle=False)
        images, _ = next(iter(loader))
        images = images.to(device)
        with torch.no_grad():
            _ = model(images)
            
        for h in merged_hooks:
            h.remove()
            
        expert_state_dict = experts[name].state_dict()
        with torch.no_grad():
            for bn_name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    mean, std = stats_merged[bn_name]
                    module.running_mean.copy_(mean)
                    module.running_var.copy_(torch.clamp(std**2 - 1e-5, min=1e-5))
                    module.weight.copy_(expert_state_dict[f"{bn_name}.weight"])
                    module.bias.copy_(expert_state_dict[f"{bn_name}.bias"])
                    
        acc = evaluate(model, test_loaders[name], experts[name])
        results[name] = acc
        
    results["average"] = np.mean([results[n] for n in datasets_list])
    return results


# --- Original Hook-based TAAC ---

def run_sequential_calibration_taac(experts, base_model, is_ta=True, lam=0.4):
    stats_orig = {}
    for name in datasets_list:
        stats_orig[name] = {}
        experts[name].eval()
        expert_hooks = []
        def make_expert_hook(bn_name):
            def hook(module, inp, out):
                x = inp[0].detach()
                mean = x.mean(dim=(0, 2, 3))
                std = torch.sqrt(x.var(dim=(0, 2, 3), unbiased=False) + 1e-5)
                stats_orig[name][bn_name] = (mean, std)
            return hook
            
        for bn_name, module in experts[name].named_modules():
            if isinstance(module, nn.BatchNorm2d):
                expert_hooks.append(module.register_forward_hook(make_expert_hook(bn_name)))
                
        loader = DataLoader(calib_datasets[name], batch_size=calibration_size, shuffle=False)
        images, _ = next(iter(loader))
        images = images.to(device)
        with torch.no_grad():
            _ = experts[name](images)
        for h in expert_hooks:
            h.remove()
            
    stats_target = {}
    first_expert = datasets_list[0]
    for bn_name in stats_orig[first_expert].keys():
        means = [stats_orig[name][bn_name][0] for name in datasets_list]
        stds = [stats_orig[name][bn_name][1] for name in datasets_list]
        stats_target[bn_name] = (torch.stack(means).mean(dim=0), torch.stack(stds).mean(dim=0))
        
    joint_images = []
    for name in datasets_list:
        loader = DataLoader(calib_datasets[name], batch_size=calibration_size, shuffle=False)
        imgs, _ = next(iter(loader))
        joint_images.append(imgs)
    joint_images_tensor = torch.cat(joint_images, dim=0)
    joint_calib_dataset = torch.utils.data.TensorDataset(joint_images_tensor, torch.zeros(len(joint_images_tensor)))
    
    if is_ta:
        model = merge_task_arithmetic(experts, base_model, lam=lam)
    else:
        model = merge_weight_averaging(experts)
    model.eval()
    
    stats_joint_merged = {}
    merged_hooks = []
    
    def make_merged_hook(bn_name):
        def hook(module, inp):
            x = inp[0]
            mean = x.mean(dim=(0, 2, 3))
            std = torch.sqrt(x.var(dim=(0, 2, 3), unbiased=False) + 1e-5)
            stats_joint_merged[bn_name] = (mean, std)
            
            mean_target, std_target = stats_target[bn_name]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
            mean_target = mean_target.view(1, -1, 1, 1)
            std_target = std_target.view(1, -1, 1, 1)
            
            x_calib = ((x - mean) / std) * std_target + mean_target
            return (x_calib,)
        return hook
        
    for bn_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            merged_hooks.append(module.register_forward_pre_hook(make_merged_hook(bn_name)))
            
    loader = DataLoader(joint_calib_dataset, batch_size=len(joint_calib_dataset), shuffle=False)
    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        _ = model(images)
        
    for h in merged_hooks:
        h.remove()
        
    with torch.no_grad():
        for bn_name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                mean, std = stats_joint_merged[bn_name]
                module.running_mean.copy_(mean)
                module.running_var.copy_(torch.clamp(std**2 - 1e-5, min=1e-5))
                
    results = {}
    for name in datasets_list:
        acc = evaluate(model, test_loaders[name], experts[name])
        results[name] = acc
    results["average"] = np.mean([results[n] for n in datasets_list])
    return results


# --- Newly Proposed Native hook-free TAAC (N-TAAC / MPNC) ---

def run_native_calibration_taac(experts, base_model, is_ta=True, lam=0.4, num_passes=1, momentum=1.0):
    # N-TAAC (Native Task-Agnostic Activation Calibration)
    if is_ta:
        model = merge_task_arithmetic(experts, base_model, lam=lam)
    else:
        model = merge_weight_averaging(experts)
        
    # Put model in train mode to enable running statistics updates
    model.train()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Set batchnorm momentum and ensure track_running_stats is True
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = momentum
            module.track_running_stats = True
            
    # Create joint calibration dataset
    joint_images = []
    for name in datasets_list:
        loader = DataLoader(calib_datasets[name], batch_size=calibration_size, shuffle=False)
        imgs, _ = next(iter(loader))
        joint_images.append(imgs)
    joint_images_tensor = torch.cat(joint_images, dim=0) # Shape: (384, 3, 32, 32)
    joint_calib_dataset = torch.utils.data.TensorDataset(joint_images_tensor, torch.zeros(len(joint_images_tensor)))
    
    # Run forward passes on the joint calibration set
    # Using batch_size=len(joint_calib_dataset) so the entire joint set is processed in a single batch
    loader = DataLoader(joint_calib_dataset, batch_size=len(joint_calib_dataset), shuffle=False)
    images, _ = next(iter(loader))
    images = images.to(device)
    
    with torch.no_grad():
        for _ in range(num_passes):
            _ = model(images)
            
    # Put model back in eval mode
    model.eval()
    
    # Reset batchnorm momentum to default (0.1) for future operations
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.1
            
    # Evaluate static task-agnostic model on all tasks
    results = {}
    for name in datasets_list:
        acc = evaluate(model, test_loaders[name], experts[name])
        results[name] = acc
    results["average"] = np.mean([results[n] for n in datasets_list])
    return results


print("\n=== Evaluating Weight Averaging (WA) Segment ===")
wa_model = merge_weight_averaging(experts)
wa_results = evaluate_multi_task(wa_model)
print(f"Weight Averaging (WA) Baseline (Uncalibrated): {wa_results}")

tcac_wa = run_sequential_calibration_tcac(experts, base_model, is_ta=False)
print(f"Weight Averaging + TCAC (Task-Conditional): {tcac_wa}")

taac_wa = run_sequential_calibration_taac(experts, base_model, is_ta=False)
print(f"Weight Averaging + TAAC (Ours, Hook-based): {taac_wa}")

print("\n--- Testing hook-free Native TAAC (N-TAAC / MPNC) on WA ---")
for passes in [1, 2, 5, 10]:
    for mom in [1.0, 0.5, 0.1]:
        # Momentum of 1.0 is only suitable for 1 pass (or acts as overwrite on each pass)
        if mom == 1.0 and passes > 1:
            continue
        n_taac_wa = run_native_calibration_taac(experts, base_model, is_ta=False, num_passes=passes, momentum=mom)
        print(f"Weight Averaging + N-TAAC (passes={passes}, momentum={mom}): {n_taac_wa}")


print("\n=== Evaluating Task Arithmetic (TA) Segment ===")
# Let's run a fine sweep of lambda for Task Arithmetic
lambda_sweep = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

for lam in lambda_sweep:
    print(f"\n--- Lambda = {lam} ---")
    ta_model = merge_task_arithmetic(experts, base_model, lam=lam)
    ta_results = evaluate_multi_task(ta_model)
    print(f"Task Arithmetic (TA) Baseline (Uncalibrated): {ta_results}")
    
    tcac_ta = run_sequential_calibration_tcac(experts, base_model, lam=lam, is_ta=True)
    print(f"Task Arithmetic + TCAC (Task-Conditional): {tcac_ta}")
    
    taac_ta = run_sequential_calibration_taac(experts, base_model, lam=lam, is_ta=True)
    print(f"Task Arithmetic + TAAC (Ours, Hook-based): {taac_ta}")
    
    n_taac_ta = run_native_calibration_taac(experts, base_model, is_ta=True, lam=lam, num_passes=1, momentum=1.0)
    print(f"Task Arithmetic + N-TAAC (Ours, Native 1-pass mom=1.0): {n_taac_ta}")
