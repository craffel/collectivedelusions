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

# Set seeds for initial setup
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

# Helper to get backbone parameters
def get_backbone_state(model):
    state = {}
    for name, param in model.state_dict().items():
        if not name.startswith("fc."):
            state[name] = param.clone()
    return state

# Load pre-trained base backbone
base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
base_backbone = get_backbone_state(base_model.to(device))

# Calibration Hook Manager
class CalibrationHookManager:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_names = []
        self.layer_dict = {}
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
        self.mode = "gather"
        self.config = "FDSA"
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

    def process_stats(self, config_type="FDSA", gamma_max=5.0):
        self.remove_hooks()
        self.mode = "active"
        self.config = config_type
        self.params = {}
        
        for name in self.layer_names:
            expert_mags = torch.cat(self.stats[name]["expert_mags"], dim=0)
            target_mag_L = expert_mags.mean(dim=(0, 1))
            target_mean_sp = torch.stack(self.stats[name]["expert_means"]).mean(dim=0).to(device)
            target_std_sp = torch.stack(self.stats[name]["expert_stds"]).mean(dim=0).to(device)
            
            target_mag_C = expert_mags.mean(dim=0)
            
            merged_mags = torch.cat(self.stats[name]["merged_mags"], dim=0)
            merged_mag_L = merged_mags.mean(dim=(0, 1))
            merged_mag_C = merged_mags.mean(dim=0)
            
            merged_mean_sp = torch.stack(self.stats[name]["merged_means"]).mean(dim=0).to(device)
            merged_std_sp = torch.stack(self.stats[name]["merged_stds"]).mean(dim=0).to(device)
            
            self.params[name] = {
                "target_mean_sp": target_mean_sp,
                "target_std_sp": target_std_sp,
                "merged_mean_sp": merged_mean_sp,
                "merged_std_sp": merged_std_sp
            }
            
            eps = 1e-5
            if config_type == "SP-TAAC":
                gamma = target_std_sp.mean() / (merged_std_sp.mean() + eps)
                self.params[name]["gamma"] = gamma
            elif config_type == "FDSA":
                Gamma = target_mag_L / (merged_mag_L + eps)
                Gamma = torch.clamp(Gamma, 1.0/gamma_max, gamma_max).to(device)
                self.params[name]["Gamma"] = Gamma
            elif config_type == "C-FDSA":
                Gamma = target_mag_C / (merged_mag_C + eps)
                Gamma = torch.clamp(Gamma, 1.0/gamma_max, gamma_max).to(device)
                self.params[name]["Gamma"] = Gamma
            elif config_type == "SMAC":
                Gamma = target_mag_L / (merged_mag_L + eps)
                Gamma = torch.clamp(Gamma, 1.0/gamma_max, gamma_max).to(device)
                self.params[name]["Gamma"] = Gamma
            elif config_type == "C-SMAC":
                Gamma = target_mag_C / (merged_mag_C + eps)
                Gamma = torch.clamp(Gamma, 1.0/gamma_max, gamma_max).to(device)
                self.params[name]["Gamma"] = Gamma

    def register_active_hooks(self):
        self.remove_hooks()
        self.mode = "active"
        
        def make_hook(name):
            p = self.params[name]
            def hook(module, input, output):
                eps = 1e-5
                if self.config == "SP-TAAC":
                    gamma = p["gamma"]
                    return output * gamma
                    
                elif self.config in ["FDSA", "C-FDSA"]:
                    Gamma = p["Gamma"]
                    x_fft = torch.fft.fft2(output, dim=(-2, -1))
                    mag = torch.abs(x_fft)
                    phase = torch.angle(x_fft)
                    scaled_mag = mag * Gamma
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
                    mean_batch = output_calibrated.mean(dim=(0, 2, 3), keepdim=True)
                    std_batch = output_calibrated.std(dim=(0, 2, 3), keepdim=True)
                    
                    target_mean = p["target_mean_sp"].view(1, -1, 1, 1)
                    target_std = p["target_std_sp"].view(1, -1, 1, 1)
                    
                    output_aligned = (output_calibrated - mean_batch) / (std_batch + eps) * target_std + target_mean
                    return output_aligned
                return output
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

def run_head_sft(merged_model, expert_heads, calib_datasets, N, epochs=25, lr=1e-3, hook_manager=None, config_type=None):
    if hook_manager is not None and config_type is not None:
        hook_manager.process_stats(config_type)
        hook_manager.register_active_hooks()
    else:
        if hook_manager is not None:
            hook_manager.remove_hooks()
            
    sft_accuracies = {}
    for task in tasks:
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(expert_heads[task])
        merged_model.fc = head
        
        cal_ds = calib_datasets[task]
        cal_loader = DataLoader(cal_ds, batch_size=min(32, N), shuffle=True)
        
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        merged_model.eval()
        head.train()
        
        for epoch in range(epochs):
            for inputs, targets in cal_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = merged_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
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
    if hook_manager is not None:
        hook_manager.remove_hooks()
    return sft_accuracies

# Sweep Parameters
backbone_experts = {t: get_backbone_state(experts[t]) for t in tasks}
expert_heads = {t: copy.deepcopy(experts[t].fc.state_dict()) for t in tasks}

calibration_budgets = [64]  # Focus on N=64 for robustness sweep
merging_algorithms = ["WA"]
seeds = [100, 101, 102] # Run 3 seeds for robust results

skew_results = {algo: {} for algo in merging_algorithms}

for algo in merging_algorithms:
    skew_results[algo]["uncalibrated_standalone"] = {t: [] for t in tasks + ["average"]}
    skew_results[algo]["uncalibrated_head_sft"] = {t: [] for t in tasks + ["average"]}
    for N in calibration_budgets:
        skew_results[algo][N] = {}
        configs = ["SP-TAAC_standalone", "FDSA_standalone", "C-FDSA_standalone", "SMAC_standalone", "C-SMAC_standalone",
                   "SP-TAAC_head_sft", "FDSA_head_sft", "C-FDSA_head_sft", "SMAC_head_sft", "C-SMAC_head_sft"]
        for cfg in configs:
            skew_results[algo][N][cfg] = {t: [] for t in tasks + ["average"]}

print(f"\nStarting Class-Skewed Sweep (only classes 0-4) over seeds {seeds}...")

for seed_idx, seed in enumerate(seeds):
    print(f"\n=========================================")
    print(f"RUNNING SKEWED SEED {seed} ({seed_idx + 1}/{len(seeds)})")
    print(f"=========================================")
    
    for algo in merging_algorithms:
        merged_model = merge_models(backbone_experts, merge_algo=algo, lambda_val=0.3)
        
        # Evaluate Uncalibrated Standalone (independent of seed, but we can evaluate once)
        base_accs = evaluate_merged(merged_model, expert_heads)
        for t in tasks + ["average"]:
            skew_results[algo]["uncalibrated_standalone"][t].append(base_accs[t])
        print(f"Uncalibrated Standalone: {base_accs['average']:.2f}% (MNIST: {base_accs['mnist']:.2f}%, FMNIST: {base_accs['fashionmnist']:.2f}%, CIFAR: {base_accs['cifar10']:.2f}%)")
            
        for N in calibration_budgets:
            print(f"--- Seed {seed} | Algo {algo} | N = {N} (Skewed) ---")
            # Prepare skewed calibration datasets for this seed and budget
            calib_datasets = {}
            expert_forward_loaders = {}
            for task in tasks:
                train_dataset = get_dataset(task, train=True)
                
                # Filter labels to only classes 0-4
                if hasattr(train_dataset, "targets"):
                    targets = train_dataset.targets
                    if not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(targets)
                else:
                    targets = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
                
                valid_indices = (targets < 5).nonzero(as_tuple=True)[0].tolist()
                
                g = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(valid_indices), generator=g).tolist()
                shuffled_valid = [valid_indices[i] for i in perm]
                
                calib_sub = Subset(train_dataset, shuffled_valid[:N])
                calib_datasets[task] = calib_sub
                expert_forward_loaders[task] = DataLoader(calib_sub, batch_size=N, shuffle=False)
                
            # Joint calibration set
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
            
            hook_manager = CalibrationHookManager(merged_model)
            
            # Gather Expert stats
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
                
            # Gather Merged stats
            hook_manager.register_gather_hooks(target_type="merged")
            with torch.no_grad():
                for inputs, _ in joint_loader:
                    inputs = inputs.to(device)
                    _ = merged_model(inputs)
                    
            # Uncalibrated + Head SFT (only evaluated at N=64)
            if N == 64:
                sft_accs = run_head_sft(merged_model, expert_heads, calib_datasets, N, epochs=25, lr=1e-3)
                for t in tasks + ["average"]:
                    skew_results[algo]["uncalibrated_head_sft"][t].append(sft_accs[t])
                print(f"Uncalibrated + Head SFT: {sft_accs['average']:.2f}%")
                    
            # Main Calibration Configs
            configs_to_run = ["SP-TAAC", "FDSA", "C-FDSA", "SMAC", "C-SMAC"]
            for config in configs_to_run:
                # 1. Standalone
                accs = evaluate_merged(merged_model, expert_heads, hook_manager, config)
                for t in tasks + ["average"]:
                    skew_results[algo][N][f"{config}_standalone"][t].append(accs[t])
                print(f"  {config} Standalone: {accs['average']:.2f}%")
                
                # 2. Head SFT
                sft_accs = run_head_sft(merged_model, expert_heads, calib_datasets, N, epochs=25, lr=1e-3, hook_manager=hook_manager, config_type=config)
                for t in tasks + ["average"]:
                    skew_results[algo][N][f"{config}_head_sft"][t].append(sft_accs[t])
                print(f"  {config} + Head SFT: {sft_accs['average']:.2f}%")

# Aggregate results
aggregated = {algo: {} for algo in merging_algorithms}
for algo in merging_algorithms:
    aggregated[algo]["uncalibrated_standalone"] = {
        t: {"mean": float(np.mean(skew_results[algo]["uncalibrated_standalone"][t])), "std": float(np.std(skew_results[algo]["uncalibrated_standalone"][t]))}
        for t in tasks + ["average"]
    }
    aggregated[algo]["uncalibrated_head_sft"] = {
        t: {"mean": float(np.mean(skew_results[algo]["uncalibrated_head_sft"][t])), "std": float(np.std(skew_results[algo]["uncalibrated_head_sft"][t]))}
        for t in tasks + ["average"]
    }
    for N in calibration_budgets:
        aggregated[algo][str(N)] = {}
        for cfg in skew_results[algo][N].keys():
            aggregated[algo][str(N)][cfg] = {
                t: {"mean": float(np.mean(skew_results[algo][N][cfg][t])), "std": float(np.std(skew_results[algo][N][cfg][t]))}
                for t in tasks + ["average"]
            }

with open("results/skew_results.json", "w") as f:
    json.dump(aggregated, f, indent=4)

print("\n--- Summary of Skewed Results (Mean \u00b1 Std) ---")
for algo in merging_algorithms:
    print(f"\nAlgorithm: {algo}")
    print(f"Uncalibrated Standalone: {aggregated[algo]['uncalibrated_standalone']['average']['mean']:.2f}%")
    for N in calibration_budgets:
        print(f"Budget N={N}:")
        for config in ["SP-TAAC", "FDSA", "C-FDSA", "SMAC", "C-SMAC"]:
            std_mean = aggregated[algo][str(N)][f"{config}_standalone"]["average"]["mean"]
            std_std = aggregated[algo][str(N)][f"{config}_standalone"]["average"]["std"]
            print(f"  {config} Standalone: {std_mean:.2f} \u00b1 {std_std:.2f}%")
