import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import os
import copy
import numpy as np
import json

# Ensure directories exist
os.makedirs("results", exist_ok=True)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu") # Run entirely on CPU
print(f"Using device: {device}")

# Preprocessing
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

transform_color = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_dataset(task, train=True):
    if task == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=train, transform=transform_gray, download=False)
    elif task == "fashionmnist":
        return torchvision.datasets.FashionMNIST(root="./data", train=train, transform=transform_gray, download=False)
    elif task == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=train, transform=transform_color, download=False)
    else:
        raise ValueError("Unknown task")

tasks = ["mnist", "fashionmnist", "cifar10"]

# Load full test sets (subset of 2000 for speed, same as main experiment)
test_loaders = {}
for task in tasks:
    ds = get_dataset(task, train=False)
    sub_indices = list(range(min(len(ds), 2000)))
    ds_sub = Subset(ds, sub_indices)
    test_loaders[task] = DataLoader(ds_sub, batch_size=256, shuffle=False, num_workers=2)

# Load experts
experts = {}
for task in tasks:
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(f"experts/{task}_expert.pth", map_location=device))
    model = model.to(device)
    model.eval()
    experts[task] = model

# Helper to get the backbone parameters of a model
def get_backbone_state(model):
    state = {}
    for name, param in model.state_dict().items():
        if not name.startswith("fc."):
            state[name] = param.clone()
    return state

# Calibration Hook Manager
class CalibrationHookManager:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_names = []
        self.layer_dict = {}
        
        # Register hooks on all 2D BatchNorm layers
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.layer_names.append(name)
                self.layer_dict[name] = module
                
        self.clear_hooks()
        
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear_hooks(self):
        self.remove_hooks()
        self.stats = {name: {"expert_mags": [], "expert_means": [], "expert_stds": [],
                             "merged_mags": [], "merged_means": [], "merged_stds": []} 
                      for name in self.layer_names}
        self.mode = "gather" # "gather" or "active"
        self.config = "SMAC"
        self.params = {} # will hold Gamma and target stats

    def register_gather_hooks(self, target_type="expert"):
        self.remove_hooks()
        self.mode = "gather"
        
        def make_hook(name):
            def hook(module, input, output):
                mag = torch.abs(torch.fft.fft2(output, dim=(-2, -1)))
                mean_sp = output.mean(dim=(0, 2, 3))
                std_sp = output.std(dim=(0, 2, 3))
                
                if target_type == "expert":
                    self.stats[name]["expert_mags"].append(mag.detach().cpu())
                    self.stats[name]["expert_means"].append(mean_sp.detach().cpu())
                    self.stats[name]["expert_stds"].append(std_sp.detach().cpu())
                else:
                    self.stats[name]["merged_mags"].append(mag.detach().cpu())
                    self.stats[name]["merged_means"].append(mean_sp.detach().cpu())
                    self.stats[name]["merged_stds"].append(std_sp.detach().cpu())
            return hook
            
        for name in self.layer_names:
            h = self.layer_dict[name].register_forward_hook(make_hook(name))
            self.hooks.append(h)

    def process_stats(self, gamma_max=5.0):
        self.remove_hooks()
        self.mode = "active"
        self.params = {}
        
        for name in self.layer_names:
            expert_mags = torch.cat(self.stats[name]["expert_mags"], dim=0)
            target_mag_L = expert_mags.mean(dim=(0, 1))
            target_mean_sp = torch.stack(self.stats[name]["expert_means"]).mean(dim=0).to(device)
            target_std_sp = torch.stack(self.stats[name]["expert_stds"]).mean(dim=0).to(device)
            
            merged_mags = torch.cat(self.stats[name]["merged_mags"], dim=0)
            merged_mag_L = merged_mags.mean(dim=(0, 1))
            
            self.params[name] = {
                "target_mean_sp": target_mean_sp,
                "target_std_sp": target_std_sp
            }
            
            eps = 1e-5
            Gamma = target_mag_L / (merged_mag_L + eps)
            Gamma = torch.clamp(Gamma, 1.0/gamma_max, gamma_max).to(device)
            self.params[name]["Gamma"] = Gamma

    def register_active_hooks(self):
        self.remove_hooks()
        self.mode = "active"
        
        def make_hook(name):
            p = self.params[name]
            def hook(module, input, output):
                eps = 1e-5
                Gamma = p["Gamma"]
                # 1. FFT Spectral alignment
                x_fft = torch.fft.fft2(output, dim=(-2, -1))
                mag = torch.abs(x_fft)
                phase = torch.angle(x_fft)
                scaled_mag = mag * Gamma
                reconstructed_fft = torch.polar(scaled_mag, phase)
                output_calibrated = torch.fft.ifft2(reconstructed_fft, dim=(-2, -1)).real
                
                # 2. Spatial Normalization and Alignment (channel-wise)
                mean_batch = output_calibrated.mean(dim=(0, 2, 3), keepdim=True)
                std_batch = output_calibrated.std(dim=(0, 2, 3), keepdim=True)
                
                target_mean = p["target_mean_sp"].view(1, -1, 1, 1)
                target_std = p["target_std_sp"].view(1, -1, 1, 1)
                
                output_aligned = (output_calibrated - mean_batch) / (std_batch + eps) * target_std + target_mean
                return output_aligned
            return hook
            
        for name in self.layer_names:
            h = self.layer_dict[name].register_forward_hook(make_hook(name))
            self.hooks.append(h)


# Function to perform Weight Averaging (WA) merging
def merge_models_wa(backbone_experts):
    merged_backbone = {}
    keys = backbone_experts["mnist"].keys()
    
    for k in keys:
        if backbone_experts["mnist"][k].dtype in [torch.float16, torch.float32, torch.float64]:
            merged_backbone[k] = torch.stack([backbone_experts[t][k] for t in tasks]).mean(dim=0)
        else:
            merged_backbone[k] = backbone_experts["mnist"][k].clone()
            
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    model_state = model.state_dict()
    for k in keys:
        model_state[k] = merged_backbone[k]
    model.load_state_dict(model_state)
    return model.to(device)


# Evaluation helper
def evaluate_merged(model, expert_heads, hook_manager, gamma_max):
    hook_manager.process_stats(gamma_max)
    hook_manager.register_active_hooks()
            
    model.eval()
    accuracies = {}
    for task in tasks:
        model.fc.load_state_dict(expert_heads[task])
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loaders[task]:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracies[task] = 100. * correct / total
        
    accuracies["average"] = sum(accuracies.values()) / 3
    hook_manager.remove_hooks()
    return accuracies


# Main sweep setup
backbone_experts = {t: get_backbone_state(experts[t]) for t in tasks}
expert_heads = {t: copy.deepcopy(experts[t].fc.state_dict()) for t in tasks}

N = 64
print(f"Preparing deterministic calibration dataset of size N = {N}")

calib_datasets = {}
expert_forward_loaders = {}
for task in tasks:
    train_dataset = get_dataset(task, train=True)
    g = torch.Generator().manual_seed(100)
    indices = torch.randperm(len(train_dataset), generator=g).tolist()
    calib_sub = Subset(train_dataset, indices[:N])
    calib_datasets[task] = calib_sub
    expert_forward_loaders[task] = DataLoader(calib_sub, batch_size=N, shuffle=False)
    
joint_inputs = []
joint_targets = []
for task in tasks:
    loader = DataLoader(calib_datasets[task], batch_size=N, shuffle=False)
    for x, y in loader:
        joint_inputs.append(x)
        joint_targets.append(y)
joint_inputs = torch.cat(joint_inputs, dim=0)
joint_targets = torch.cat(joint_targets, dim=0)
joint_dataset = TensorDataset(joint_inputs, joint_targets)
joint_loader = DataLoader(joint_dataset, batch_size=len(joint_dataset), shuffle=False)

# Merge model
merged_model = merge_models_wa(backbone_experts)
hook_manager = CalibrationHookManager(merged_model)

# Gather Expert Statistics
print("Gathering expert statistics...")
for task in tasks:
    expert_model = experts[task]
    expert_hook_manager = CalibrationHookManager(expert_model)
    expert_hook_manager.register_gather_hooks(target_type="expert")
    
    with torch.no_grad():
        for inputs, _ in expert_forward_loaders[task]:
            inputs = inputs.to(device)
            _ = expert_model(inputs)
            
    for name in hook_manager.layer_names:
        hook_manager.stats[name]["expert_mags"].extend(expert_hook_manager.stats[name]["expert_mags"])
        hook_manager.stats[name]["expert_means"].extend(expert_hook_manager.stats[name]["expert_means"])
        hook_manager.stats[name]["expert_stds"].extend(expert_hook_manager.stats[name]["expert_stds"])
    expert_hook_manager.clear_hooks()

# Gather Merged Statistics
print("Gathering merged statistics...")
hook_manager.register_gather_hooks(target_type="merged")
with torch.no_grad():
    for inputs, _ in joint_loader:
        inputs = inputs.to(device)
        _ = merged_model(inputs)

gamma_values = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
sweep_results = {}

print("\n--- Sweeping gamma_max ---")
for g_val in gamma_values:
    # We deepcopy hook manager stats to avoid issues
    hm_temp = CalibrationHookManager(merged_model)
    for name in hook_manager.layer_names:
        hm_temp.stats[name]["expert_mags"] = list(hook_manager.stats[name]["expert_mags"])
        hm_temp.stats[name]["expert_means"] = list(hook_manager.stats[name]["expert_means"])
        hm_temp.stats[name]["expert_stds"] = list(hook_manager.stats[name]["expert_stds"])
        hm_temp.stats[name]["merged_mags"] = list(hook_manager.stats[name]["merged_mags"])
        
    accs = evaluate_merged(merged_model, expert_heads, hm_temp, g_val)
    print(f"gamma_max = {g_val:.1f} | Average: {accs['average']:.2f}% (MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fashionmnist']:.2f}%, CIFAR: {accs['cifar10']:.2f}%)")
    sweep_results[g_val] = accs

with open("results/gamma_sweep.json", "w") as f:
    json.dump(sweep_results, f, indent=4)
print("\nSweep complete! Saved to results/gamma_sweep.json")
