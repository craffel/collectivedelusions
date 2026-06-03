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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# Load full test sets
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

# Store baseline expert accuracies
oracle_accs = {}
print("\n--- Evaluating Expert Oracles ---")
for task in tasks:
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loaders[task]:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = experts[task](inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    oracle_accs[task] = acc
    print(f"Oracle {task.upper()} Accuracy: {acc:.2f}%")
oracle_accs["average"] = sum(oracle_accs.values()) / 3
print(f"Oracle Average: {oracle_accs['average']:.2f}%")


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
        self.config = "FDSA" # "SP-TAAC", "FDSA", "SMAC", "C-FDSA", "C-SMAC"
        self.params = {} # will hold gamma or Gamma, and target stats

    def register_gather_hooks(self, target_type="expert"):
        self.remove_hooks()
        self.mode = "gather"
        
        def make_hook(name):
            def hook(module, input, output):
                # output shape: [B, C, H, W]
                # FFT Magnitude
                mag = torch.abs(torch.fft.fft2(output, dim=(-2, -1))) # [B, C, H, W]
                
                # Spatial Statistics
                mean_sp = output.mean(dim=(0, 2, 3)) # [C]
                std_sp = output.std(dim=(0, 2, 3)) # [C]
                
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

    def process_stats(self, config_type="FDSA", gamma_max=5.0):
        self.remove_hooks()
        self.mode = "active"
        self.config = config_type
        self.params = {}
        
        for name in self.layer_names:
            # 1. Process Expert Stats (Target)
            # expert_mags lists: items of shape [B, C, H, W]
            expert_mags = torch.cat(self.stats[name]["expert_mags"], dim=0) # [D*N, C, H, W]
            # Average across batch dimension (first)
            # L-FDSA (Layer-wise): average across both batch and channel
            # C-FDSA (Channel-wise): average across batch only
            
            # Layer-wise (L)
            target_mag_L = expert_mags.mean(dim=(0, 1)) # [H, W]
            target_mean_sp = torch.stack(self.stats[name]["expert_means"]).mean(dim=0).to(device) # [C]
            target_std_sp = torch.stack(self.stats[name]["expert_stds"]).mean(dim=0).to(device) # [C]
            
            # Channel-wise (C)
            target_mag_C = expert_mags.mean(dim=0) # [C, H, W]
            
            # 2. Process Merged Stats (Uncalibrated)
            merged_mags = torch.cat(self.stats[name]["merged_mags"], dim=0) # [D*N, C, H, W]
            merged_mag_L = merged_mags.mean(dim=(0, 1)) # [H, W]
            merged_mag_C = merged_mags.mean(dim=0) # [C, H, W]
            
            merged_mean_sp = torch.stack(self.stats[name]["merged_means"]).mean(dim=0).to(device) # [C]
            merged_std_sp = torch.stack(self.stats[name]["merged_stds"]).mean(dim=0).to(device) # [C]
            
            # Store parameters
            self.params[name] = {
                "target_mean_sp": target_mean_sp,
                "target_std_sp": target_std_sp,
                "merged_mean_sp": merged_mean_sp,
                "merged_std_sp": merged_std_sp
            }
            
            eps = 1e-5
            if config_type == "SP-TAAC":
                # Spatial layer-wise scaling
                gamma = target_std_sp.mean() / (merged_std_sp.mean() + eps)
                self.params[name]["gamma"] = gamma
            elif config_type == "FDSA":
                # Layer-wise Spectral scaling map
                Gamma = target_mag_L / (merged_mag_L + eps)
                Gamma = torch.clamp(Gamma, 1.0/gamma_max, gamma_max).to(device)
                self.params[name]["Gamma"] = Gamma
            elif config_type == "C-FDSA":
                # Channel-wise Spectral scaling map
                Gamma = target_mag_C / (merged_mag_C + eps)
                Gamma = torch.clamp(Gamma, 1.0/gamma_max, gamma_max).to(device)
                self.params[name]["Gamma"] = Gamma
            elif config_type == "SMAC":
                # L-SMAC: L-FDSA + Spatial Alignment
                Gamma = target_mag_L / (merged_mag_L + eps)
                Gamma = torch.clamp(Gamma, 1.0/gamma_max, gamma_max).to(device)
                self.params[name]["Gamma"] = Gamma
            elif config_type == "C-SMAC":
                # C-SMAC: C-FDSA + Spatial Alignment
                Gamma = target_mag_C / (merged_mag_C + eps)
                Gamma = torch.clamp(Gamma, 1.0/gamma_max, gamma_max).to(device)
                self.params[name]["Gamma"] = Gamma

    def register_active_hooks(self):
        self.remove_hooks()
        self.mode = "active"
        
        def make_hook(name):
            p = self.params[name]
            
            def hook(module, input, output):
                # output shape: [B, C, H, W]
                eps = 1e-5
                
                if self.config == "SP-TAAC":
                    gamma = p["gamma"]
                    return output * gamma
                    
                elif self.config in ["FDSA", "C-FDSA"]:
                    Gamma = p["Gamma"] # [H, W] or [C, H, W]
                    # Compute FFT
                    x_fft = torch.fft.fft2(output, dim=(-2, -1))
                    mag = torch.abs(x_fft)
                    phase = torch.angle(x_fft)
                    # Scale magnitude
                    # For L-FDSA: Gamma shape is [H, W], broadcasted to [B, C, H, W]
                    # For C-FDSA: Gamma shape is [C, H, W], broadcasted to [B, C, H, W]
                    scaled_mag = mag * Gamma
                    # Reconstruct
                    reconstructed_fft = torch.polar(scaled_mag, phase)
                    output_calibrated = torch.fft.ifft2(reconstructed_fft, dim=(-2, -1)).real
                    return output_calibrated
                    
                elif self.config in ["SMAC", "C-SMAC"]:
                    Gamma = p["Gamma"]
                    # 1. FFT Spectral alignment
                    x_fft = torch.fft.fft2(output, dim=(-2, -1))
                    mag = torch.abs(x_fft)
                    phase = torch.angle(x_fft)
                    scaled_mag = mag * Gamma
                    reconstructed_fft = torch.polar(scaled_mag, phase)
                    output_calibrated = torch.fft.ifft2(reconstructed_fft, dim=(-2, -1)).real
                    
                    # 2. Spatial Normalization and Alignment (channel-wise)
                    # We compute statistics on the active batch to perform alignment
                    mean_batch = output_calibrated.mean(dim=(0, 2, 3), keepdim=True)
                    std_batch = output_calibrated.std(dim=(0, 2, 3), keepdim=True)
                    
                    # Target statistics
                    target_mean = p["target_mean_sp"].view(1, -1, 1, 1)
                    target_std = p["target_std_sp"].view(1, -1, 1, 1)
                    
                    # Align statistics
                    output_aligned = (output_calibrated - mean_batch) / (std_batch + eps) * target_std + target_mean
                    return output_aligned
                    
                return output
            return hook
            
        for name in self.layer_names:
            h = self.layer_dict[name].register_forward_hook(make_hook(name))
            self.hooks.append(h)


# Function to perform merging
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
            
    # Load into a new model
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    # Load backbone
    model_state = model.state_dict()
    for k in keys:
        model_state[k] = merged_backbone[k]
    # Keep expert heads (we copy them dynamically during evaluation or fine-tuning)
    model.load_state_dict(model_state)
    return model.to(device)


# Evaluation helper
def evaluate_merged(model, expert_heads, hook_manager=None, config_type=None):
    if hook_manager is not None and config_type is not None:
        hook_manager.process_stats(config_type)
        hook_manager.register_active_hooks()
    else:
        if hook_manager is not None:
            hook_manager.remove_hooks()
            
    model.eval()
    accuracies = {}
    for task in tasks:
        # Load the task-specific head
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


# Perform Head SFT
def run_head_sft(merged_model, expert_heads, calib_datasets, N, epochs=20, lr=1e-3, hook_manager=None, config_type=None):
    if hook_manager is not None and config_type is not None:
        hook_manager.process_stats(config_type)
        hook_manager.register_active_hooks()
    else:
        if hook_manager is not None:
            hook_manager.remove_hooks()
            
    # Freeze backbone
    for name, param in merged_model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
            
    sft_accuracies = {}
    
    # We fine-tune each task head separately using its calibration dataset
    for task in tasks:
        # Load expert head initial state
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(expert_heads[task])
        merged_model.fc = head
        
        # Prepare calibration dataloader
        cal_ds = calib_datasets[task]
        cal_loader = DataLoader(cal_ds, batch_size=min(32, N), shuffle=True)
        
        # Only optimize the fc head parameters
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        merged_model.eval() # Backbone remains in eval mode
        head.train()
        
        for epoch in range(epochs):
            for inputs, targets in cal_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = merged_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
        # Evaluate task
        merged_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loaders[task]:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = merged_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        sft_accuracies[task] = 100. * correct / total
        
    sft_accuracies["average"] = sum(sft_accuracies.values()) / 3
    
    # Unfreeze backbone
    for param in merged_model.parameters():
        param.requires_grad = True
        
    if hook_manager is not None:
        hook_manager.remove_hooks()
        
    return sft_accuracies


# Main Sweep Loop
backbone_experts = {t: get_backbone_state(experts[t]) for t in tasks}
expert_heads = {t: copy.deepcopy(experts[t].fc.state_dict()) for t in tasks}

calibration_budgets = [16, 64, 128]
merging_algorithms = ["WA", "TA"]

all_results = {}

for algo in merging_algorithms:
    all_results[algo] = {}
    print(f"\n=========================================")
    print(f"MERGING ALGORITHM: {algo}")
    print(f"=========================================")
    
    # Create merged model
    merged_model = merge_models(backbone_experts, merge_algo=algo, lambda_val=0.3)
    
    # 1. Uncalibrated Baseline
    print("\n--- Running Uncalibrated Baseline ---")
    base_accs = evaluate_merged(merged_model, expert_heads)
    print(f"Uncalibrated Standalone: {base_accs['average']:.2f}% (MNIST: {base_accs['mnist']:.2f}%, FMNIST: {base_accs['fashionmnist']:.2f}%, CIFAR: {base_accs['cifar10']:.2f}%)")
    all_results[algo]["uncalibrated_standalone"] = base_accs
    
    for N in calibration_budgets:
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Calibration Budget N = {N}")
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        all_results[algo][N] = {}
        
        # Prepare deterministic calibration datasets of size N
        # (We use training sets, but with a fixed deterministic split)
        calib_datasets = {}
        expert_forward_loaders = {}
        for task in tasks:
            train_dataset = get_dataset(task, train=True)
            g = torch.Generator().manual_seed(100) # different seed from training
            indices = torch.randperm(len(train_dataset), generator=g).tolist()
            # Select calibration subset
            calib_sub = Subset(train_dataset, indices[:N])
            calib_datasets[task] = calib_sub
            expert_forward_loaders[task] = DataLoader(calib_sub, batch_size=N, shuffle=False)
            
        # Joint calibration set
        # Re-batch to run forward pass of merged model on the D * N dataset
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
        
        # Instantiate Hook Manager
        hook_manager = CalibrationHookManager(merged_model)
        
        # Collect statistics
        # 1. Expert statistics
        for task in tasks:
            expert_model = experts[task]
            expert_hook_manager = CalibrationHookManager(expert_model)
            expert_hook_manager.register_gather_hooks(target_type="expert")
            
            # Run forward pass of expert
            with torch.no_grad():
                for inputs, _ in expert_forward_loaders[task]:
                    inputs = inputs.to(device)
                    _ = expert_model(inputs)
                    
            # Move expert gathered stats to merged model hook manager
            for name in hook_manager.layer_names:
                hook_manager.stats[name]["expert_mags"].extend(expert_hook_manager.stats[name]["expert_mags"])
                hook_manager.stats[name]["expert_means"].extend(expert_hook_manager.stats[name]["expert_means"])
                hook_manager.stats[name]["expert_stds"].extend(expert_hook_manager.stats[name]["expert_stds"])
            expert_hook_manager.clear_hooks()
            
        # 2. Merged statistics (Uncalibrated)
        hook_manager.register_gather_hooks(target_type="merged")
        with torch.no_grad():
            for inputs, _ in joint_loader:
                inputs = inputs.to(device)
                _ = merged_model(inputs)
                
        # Now run evaluations for each calibration configuration
        configs = ["SP-TAAC", "FDSA", "C-FDSA", "SMAC", "C-SMAC"]
        
        # First, uncalibrated + head SFT (at N=64)
        if N == 64:
            print("\n--- Running Uncalibrated + Head SFT ---")
            sft_accs = run_head_sft(merged_model, expert_heads, calib_datasets, N, epochs=25, lr=1e-3)
            print(f"Uncalibrated + Head SFT: {sft_accs['average']:.2f}% (MNIST: {sft_accs['mnist']:.2f}%, FMNIST: {sft_accs['fashionmnist']:.2f}%, CIFAR: {sft_accs['cifar10']:.2f}%)")
            all_results[algo]["uncalibrated_head_sft"] = sft_accs
            
        for config in configs:
            print(f"\n--- Method: {config} ---")
            
            # Evaluate Standalone Calibration
            accs = evaluate_merged(merged_model, expert_heads, hook_manager, config)
            print(f"{config} Standalone: {accs['average']:.2f}% (MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fashionmnist']:.2f}%, CIFAR: {accs['cifar10']:.2f}%)")
            all_results[algo][N][f"{config}_standalone"] = accs
            
            # Evaluate Calibration + Head SFT (at N=64)
            if N == 64:
                # We need to compute hook params again because run_head_sft clears hooks, but run_head_sft handles this internally!
                sft_accs = run_head_sft(merged_model, expert_heads, calib_datasets, N, epochs=25, lr=1e-3, hook_manager=hook_manager, config_type=config)
                print(f"{config} + Head SFT: {sft_accs['average']:.2f}% (MNIST: {sft_accs['mnist']:.2f}%, FMNIST: {sft_accs['fashionmnist']:.2f}%, CIFAR: {sft_accs['cifar10']:.2f}%)")
                all_results[algo][N][f"{config}_head_sft"] = sft_accs

# Save all results to a JSON file
with open("results/all_results.json", "w") as f:
    json.dump(all_results, f, indent=4)
print("\nSaved all results to results/all_results.json")
