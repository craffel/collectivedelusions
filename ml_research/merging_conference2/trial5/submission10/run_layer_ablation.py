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

device = torch.device("cpu")
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

# Load full test sets (subset of 2000 for speed)
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

# Load pre-trained base backbone (ImageNet)
base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
base_backbone = get_backbone_state(base_model.to(device))

# Calibration Hook Manager
class CalibrationHookManager:
    def __init__(self, model, subset="all"):
        self.model = model
        self.hooks = []
        self.layer_names = []
        self.layer_dict = {}
        self.subset = subset
        
        # Register hooks on all or a subset of 2D BatchNorm layers
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if subset == "all":
                    self.layer_names.append(name)
                elif subset == "early":
                    if name in ["bn1", "layer1.0.bn1", "layer1.0.bn2", "layer1.1.bn1", "layer1.1.bn2"]:
                        self.layer_names.append(name)
                elif subset == "middle":
                    if any(x in name for x in ["layer2", "layer3"]):
                        self.layer_names.append(name)
                elif subset == "late":
                    if "layer4" in name:
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
        self.mode = "gather"
        self.config = "SMAC"
        self.params = {}

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
            merged_mean_sp = torch.stack(self.stats[name]["merged_means"]).mean(dim=0).to(device)
            merged_std_sp = torch.stack(self.stats[name]["merged_stds"]).mean(dim=0).to(device)
            
            self.params[name] = {
                "target_mean_sp": target_mean_sp,
                "target_std_sp": target_std_sp,
                "merged_mean_sp": merged_mean_sp,
                "merged_std_sp": merged_std_sp
            }
            
            eps = 1e-5
            # L-SMAC (SMAC) logic
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

def merge_models(backbone_experts, merge_algo="WA", lambda_val=0.3):
    merged_backbone = {}
    keys = backbone_experts["mnist"].keys()
    
    if merge_algo == "WA":
        for k in keys:
            if backbone_experts["mnist"][k].dtype in [torch.float16, torch.float32, torch.float64]:
                merged_backbone[k] = torch.stack([backbone_experts[t][k] for t in tasks]).mean(dim=0)
            else:
                merged_backbone[k] = backbone_experts["mnist"][k].clone()
    elif merge_algo == "TA":
        for k in keys:
            if backbone_experts["mnist"][k].dtype in [torch.float16, torch.float32, torch.float64]:
                task_vectors = [backbone_experts[t][k] - base_backbone[k] for t in tasks]
                merged_backbone[k] = base_backbone[k] + lambda_val * sum(task_vectors)
            else:
                merged_backbone[k] = backbone_experts["mnist"][k].clone()
                
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    model_state = model.state_dict()
    for k in keys:
        model_state[k] = merged_backbone[k]
    model.load_state_dict(model_state)
    return model.to(device)

def evaluate_merged_subset(model, expert_heads, hook_manager=None):
    if hook_manager is not None:
        hook_manager.process_stats()
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
    if hook_manager is not None:
        hook_manager.remove_hooks()
    return accuracies

# Parameters
backbone_experts = {t: get_backbone_state(experts[t]) for t in tasks}
expert_heads = {t: copy.deepcopy(experts[t].fc.state_dict()) for t in tasks}

N = 64
merging_algorithms = ["WA", "TA"]
subsets = ["all", "early", "middle", "late"]
seeds = [100, 101, 102]

raw_results = {algo: {subset: {t: [] for t in tasks + ["average"]} for subset in subsets} for algo in merging_algorithms}

print(f"\nStarting Layer-wise Ablation Sweep...")

for seed in seeds:
    print(f"\n--- Seed {seed} ---")
    for algo in merging_algorithms:
        print(f"  Merging Algorithm: {algo}")
        merged_model = merge_models(backbone_experts, merge_algo=algo, lambda_val=0.3)
        
        # Prepare calibration sets
        calib_datasets = {}
        expert_forward_loaders = {}
        for task in tasks:
            train_dataset = get_dataset(task, train=True)
            g = torch.Generator().manual_seed(seed)
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
        
        for subset in subsets:
            hook_manager = CalibrationHookManager(merged_model, subset=subset)
            
            # Gather Expert stats
            for task in tasks:
                expert_model = experts[task]
                expert_hook_manager = CalibrationHookManager(expert_model, subset=subset)
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
                
            # Gather Merged stats
            hook_manager.register_gather_hooks(target_type="merged")
            with torch.no_grad():
                for inputs, _ in joint_loader:
                    inputs = inputs.to(device)
                    _ = merged_model(inputs)
                    
            # Evaluate
            accs = evaluate_merged_subset(merged_model, expert_heads, hook_manager)
            print(f"    Subset: {subset:<8} | Acc: {accs['average']:.2f}% (MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fashionmnist']:.2f}%, CIFAR: {accs['cifar10']:.2f}%)")
            for t in tasks + ["average"]:
                raw_results[algo][subset][t].append(accs[t])

# Process statistics
final_stats = {algo: {} for algo in merging_algorithms}
for algo in merging_algorithms:
    for subset in subsets:
        final_stats[algo][subset] = {}
        for t in tasks + ["average"]:
            arr = np.array(raw_results[algo][subset][t])
            final_stats[algo][subset][t] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr))
            }

with open("results/layer_ablation_results.json", "w") as f:
    json.dump(final_stats, f, indent=4)
print("\nSaved layer ablation results to results/layer_ablation_results.json")
