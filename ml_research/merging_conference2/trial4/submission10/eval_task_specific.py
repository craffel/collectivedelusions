import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import copy
import numpy as np

# Set random seeds and disable cuDNN for stability
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

def get_transforms(dataset_name):
    if dataset_name in ["mnist", "fmnist"]:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else: # cifar10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform

def get_dataset(name, train=True):
    transform = get_transforms(name)
    if name == "mnist":
        dataset = datasets.MNIST(root="./data", train=train, download=False, transform=transform)
    elif name == "fmnist":
        dataset = datasets.FashionMNIST(root="./data", train=train, download=False, transform=transform)
    elif name == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=train, download=False, transform=transform)
    return dataset

# 1. Load Expert Models and classification heads
def load_experts(device):
    datasets_list = ["mnist", "fmnist", "cifar10"]
    experts = {}
    heads = {}
    
    # Load base model to get the state dict structure
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    
    for ds in datasets_list:
        model = copy.deepcopy(base_model)
        model.load_state_dict(torch.load(f"weights/expert_{ds}.pth", map_location=device))
        model = model.to(device)
        model.eval()
        experts[ds] = model
        # Extract classification head
        heads[ds] = copy.deepcopy(model.fc)
        
    return experts, heads, base_model.to(device)

# 2. Perform Merging
def merge_models(experts, base_model, method="wa", lam=0.3):
    merged_model = copy.deepcopy(base_model)
    merged_state = merged_model.state_dict()
    
    expert_states = {k: v.state_dict() for k, v in experts.items()}
    base_state = base_model.state_dict()
    
    # We do not merge the 'fc' classification heads
    keys_to_merge = [k for k in merged_state.keys() if "fc" not in k]
    
    if method == "wa":
        for key in keys_to_merge:
            tensors = [expert_states[ds][key] for ds in experts]
            if tensors[0].dtype.is_floating_point or tensors[0].dtype.is_complex:
                merged_state[key] = torch.stack(tensors).mean(dim=0)
            else:
                merged_state[key] = tensors[0]
    elif method == "ta":
        for key in keys_to_merge:
            tensors = [expert_states[ds][key] for ds in experts]
            if tensors[0].dtype.is_floating_point or tensors[0].dtype.is_complex:
                task_vectors = [expert_states[ds][key] - base_state[key] for ds in experts]
                merged_state[key] = base_state[key] + lam * sum(task_vectors)
            else:
                merged_state[key] = base_state[key]
            
    merged_model.load_state_dict(merged_state)
    return merged_model

def clip_scale(scale, max_val=5.0):
    return torch.clamp(scale, min=1.0/max_val, max=max_val)


# ==================== Task-Specific Hooks ====================

class TS_SPTAAC_Hook:
    """ Task-Specific Layer-wise Spatial Scaling """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_stds = {}  # task -> std
        self.merged_stds = {}  # task -> std
        self.gammas = {}       # task -> gamma scalar
        self.current_task = None
        self.expert_std_temp = None
        self.merged_std_temp = None

    def start_expert_calibration(self):
        self.calibrating_target = True

    def save_expert_std(self, std):
        self.expert_std_temp = std

    def finalize_target(self, ds):
        self.calibrating_target = False
        self.expert_stds[ds] = self.expert_std_temp

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, ds, std):
        self.calibrating_merged = False
        self.merged_stds[ds] = std
        self.gammas[ds] = self.expert_stds[ds].to(std.device) / (std + 1e-5)
        self.gammas[ds] = clip_scale(self.gammas[ds])
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            channel_stds = torch.sqrt(torch.var(output, dim=(0, 2, 3), unbiased=False) + 1e-5)
            layer_std = torch.mean(channel_stds)
            self.save_expert_std(layer_std.detach().cpu())
            return output
            
        if self.calibrating_merged:
            channel_stds = torch.sqrt(torch.var(output, dim=(0, 2, 3), unbiased=False) + 1e-5)
            layer_std = torch.mean(channel_stds)
            self.merged_std_temp = layer_std.detach().cpu()
            return output
            
        if self.active and self.current_task is not None:
            g = self.gammas[self.current_task].to(output.device)
            return output * g
            
        return output


class TS_TCAC_Hook:
    """ Task-Specific Channel-wise Spatial Scaling """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_stds = {}  # task -> std
        self.merged_stds = {}  # task -> std
        self.gammas = {}       # task -> gamma vector
        self.current_task = None
        self.expert_std_temp = None
        self.merged_std_temp = None

    def start_expert_calibration(self):
        self.calibrating_target = True

    def save_expert_std(self, std):
        self.expert_std_temp = std

    def finalize_target(self, ds):
        self.calibrating_target = False
        self.expert_stds[ds] = self.expert_std_temp

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, ds, std):
        self.calibrating_merged = False
        self.merged_stds[ds] = std
        self.gammas[ds] = self.expert_stds[ds].to(std.device) / (std + 1e-5)
        self.gammas[ds] = clip_scale(self.gammas[ds])
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            std = torch.sqrt(torch.var(output, dim=(0, 2, 3), unbiased=False) + 1e-5) # shape (C,)
            self.save_expert_std(std.detach().cpu())
            return output
            
        if self.calibrating_merged:
            std = torch.sqrt(torch.var(output, dim=(0, 2, 3), unbiased=False) + 1e-5)
            self.merged_std_temp = std.detach().cpu()
            return output
            
        if self.active and self.current_task is not None:
            g = self.gammas[self.current_task].to(output.device)
            return output * g.view(1, -1, 1, 1)
            
        return output


class TS_LFDSA_Hook:
    """ Task-Specific Layer-wise Frequency-Domain Spectral Alignment """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_mags = {}  # task -> 2D magnitude map
        self.merged_mags = {}  # task -> 2D magnitude map
        self.gammas = {}       # task -> 2D scaling map
        self.current_task = None
        self.expert_mag_temp = None
        self.merged_mag_temp = None

    def start_expert_calibration(self):
        self.calibrating_target = True

    def finalize_target(self, ds):
        self.calibrating_target = False
        self.expert_mags[ds] = self.expert_mag_temp

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, ds, mag):
        self.calibrating_merged = False
        self.merged_mags[ds] = mag
        self.gammas[ds] = self.expert_mags[ds].to(mag.device) / (mag + 1e-5)
        self.gammas[ds] = clip_scale(self.gammas[ds])
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=(0, 1)) # shape (H, W)
            self.expert_mag_temp = mean_mag.detach().cpu()
            return output
            
        if self.calibrating_merged:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=(0, 1)) # shape (H, W)
            self.merged_mag_temp = mean_mag.detach().cpu()
            return output
            
        if self.active and self.current_task is not None:
            g = self.gammas[self.current_task].to(output.device)
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            scaled_fft = fft_out * g.view(1, 1, g.shape[0], g.shape[1])
            calibrated_out = torch.real(torch.fft.ifft2(scaled_fft, dim=(-2, -1)))
            return calibrated_out
            
        return output


class TS_CFDSA_Hook:
    """ Task-Specific Channel-wise Frequency-Domain Spectral Alignment """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_mags = {}  # task -> 3D magnitude map (C, H, W)
        self.merged_mags = {}  # task -> 3D magnitude map (C, H, W)
        self.gammas = {}       # task -> 3D scaling map
        self.current_task = None
        self.expert_mag_temp = None
        self.merged_mag_temp = None

    def start_expert_calibration(self):
        self.calibrating_target = True

    def finalize_target(self, ds):
        self.calibrating_target = False
        self.expert_mags[ds] = self.expert_mag_temp

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, ds, mag):
        self.calibrating_merged = False
        self.merged_mags[ds] = mag
        self.gammas[ds] = self.expert_mags[ds].to(mag.device) / (mag + 1e-5)
        self.gammas[ds] = clip_scale(self.gammas[ds])
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=0) # shape (C, H, W)
            self.expert_mag_temp = mean_mag.detach().cpu()
            return output
            
        if self.calibrating_merged:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=0) # shape (C, H, W)
            self.merged_mag_temp = mean_mag.detach().cpu()
            return output
            
        if self.active and self.current_task is not None:
            g = self.gammas[self.current_task].to(output.device)
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            scaled_fft = fft_out * g.unsqueeze(0) # shape (1, C, H, W)
            calibrated_out = torch.real(torch.fft.ifft2(scaled_fft, dim=(-2, -1)))
            return calibrated_out
            
        return output


# ==================== Registration and Orchestration ====================

def register_calibration_hooks(model, hook_class):
    hooks = []
    hook_objs = []
    for name, module in model.named_modules():
        # Match batch normalization layers, which are standard for ResNet calibration
        if isinstance(module, nn.BatchNorm2d):
            hook_obj = hook_class()
            hook_fn = hook_obj.hook_fn
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
            hook_objs.append(hook_obj)
    return hooks, hook_objs

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

def run_task_specific_calibration(model, experts, heads, datasets_list, N, device, calibration_type="ts-l-fdsa"):
    if calibration_type == "ts-sp-taac":
        hooks, hook_objs = register_calibration_hooks(model, TS_SPTAAC_Hook)
    elif calibration_type == "ts-tcac":
        hooks, hook_objs = register_calibration_hooks(model, TS_TCAC_Hook)
    elif calibration_type == "ts-l-fdsa":
        hooks, hook_objs = register_calibration_hooks(model, TS_LFDSA_Hook)
    elif calibration_type == "ts-c-fdsa":
        hooks, hook_objs = register_calibration_hooks(model, TS_CFDSA_Hook)
    else:
        raise ValueError(f"Unknown task-specific calibration type {calibration_type}")

    # Phase 1: Collect task-specific statistics from experts and store them directly
    for i, ds in enumerate(datasets_list):
        train_dataset = get_dataset(ds, train=True)
        generator = torch.Generator().manual_seed(100 + i)
        calib_indices = torch.randperm(len(train_dataset), generator=generator)[:N].tolist()
        calib_subset = Subset(train_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=N, shuffle=False)
        
        expert = experts[ds]
        if calibration_type == "ts-sp-taac":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, TS_SPTAAC_Hook)
        elif calibration_type == "ts-tcac":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, TS_TCAC_Hook)
        elif calibration_type == "ts-l-fdsa":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, TS_LFDSA_Hook)
        elif calibration_type == "ts-c-fdsa":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, TS_CFDSA_Hook)
            
        for h_obj in exp_hook_objs:
            h_obj.start_expert_calibration()
            
        inputs, _ = next(iter(calib_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = expert(inputs)
            
        # Copy the task-specific statistic to the main hooks
        for exp_h_obj, main_h_obj in zip(exp_hook_objs, hook_objs):
            if calibration_type in ["ts-sp-taac", "ts-tcac"]:
                main_h_obj.expert_stds[ds] = exp_h_obj.expert_std_temp
            else:
                main_h_obj.expert_mags[ds] = exp_h_obj.expert_mag_temp
                
        remove_hooks(exp_hooks)

    # Phase 2: Collect merged model statistics on each task-specific dataset separately
    for i, ds in enumerate(datasets_list):
        train_dataset = get_dataset(ds, train=True)
        generator = torch.Generator().manual_seed(100 + i)
        calib_indices = torch.randperm(len(train_dataset), generator=generator)[:N].tolist()
        calib_subset = Subset(train_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=N, shuffle=False)
        inputs, _ = next(iter(calib_loader))
        inputs = inputs.to(device)
        
        # Set main hooks to calibrating merged mode
        for main_h_obj in hook_objs:
            main_h_obj.start_merged_calibration()
            
        # Run forward pass of the merged model on this task's calibration data
        model.fc = copy.deepcopy(heads[ds]).to(device)
        with torch.no_grad():
            _ = model(inputs)
            
        # Finalize statistics for this specific task
        for main_h_obj in hook_objs:
            if calibration_type in ["ts-sp-taac", "ts-tcac"]:
                main_h_obj.finalize_merged(ds, main_h_obj.merged_std_temp)
            else:
                main_h_obj.finalize_merged(ds, main_h_obj.merged_mag_temp)

    return hooks, hook_objs

def evaluate_merged_task_specific(model, heads, hook_objs, device):
    datasets_list = ["mnist", "fmnist", "cifar10"]
    accuracies = {}
    
    for ds in datasets_list:
        test_dataset = get_dataset(ds, train=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
        
        # Set active task on all hooks
        for h in hook_objs:
            h.current_task = ds
            
        # Swap head
        model.fc = copy.deepcopy(heads[ds]).to(device)
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        acc = 100.0 * correct / total
        accuracies[ds] = acc
        
    accuracies["average"] = sum(accuracies.values()) / len(datasets_list)
    return accuracies

def run_head_sft_task_specific(model, heads, datasets_list, N, hook_objs, device, epochs=15):
    adapted_heads = {}
    model.eval()
    
    for i, ds in enumerate(datasets_list):
        # Set task on all hooks
        for h in hook_objs:
            h.current_task = ds
            
        train_dataset = get_dataset(ds, train=True)
        generator = torch.Generator().manual_seed(100 + i)
        calib_indices = torch.randperm(len(train_dataset), generator=generator)[:N].tolist()
        calib_subset = Subset(train_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=32, shuffle=True)
        
        # Clone expert classification head
        adapted_head = copy.deepcopy(heads[ds]).to(device)
        adapted_head.train()
        
        optimizer = torch.optim.AdamW(adapted_head.parameters(), lr=1e-3, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for inputs, labels in calib_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                # Extract features through the backbone with hooks active
                with torch.no_grad():
                    model.fc = nn.Identity()
                    features = model(inputs)
                    
                outputs = adapted_head(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        adapted_heads[ds] = adapted_head.eval()
        
    return adapted_heads


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load expert models and heads
    experts, heads, base_model = load_experts(device)
    datasets_list = ["mnist", "fmnist", "cifar10"]
    
    results = {}
    
    # 1. Oracle Experts alone
    print("\nEvaluating Individual Expert Oracles...")
    oracle_accs = {}
    # To evaluate experts alone we don't need hooks
    for ds in datasets_list:
        # Evaluate experts using a quick dummy function
        test_dataset = get_dataset(ds, train=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
        expert = experts[ds]
        expert.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = expert(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        oracle_accs[ds] = 100.0 * correct / total
    oracle_accs["average"] = sum(oracle_accs.values()) / len(datasets_list)
    results["Oracle"] = oracle_accs
    print(f"Oracle: MNIST: {oracle_accs['mnist']:.2f}%, FMNIST: {oracle_accs['fmnist']:.2f}%, CIFAR10: {oracle_accs['cifar10']:.2f}%, Avg: {oracle_accs['average']:.2f}%")

    # Evaluate Task-Specific Merging Configurations
    for merge_method in ["wa", "ta"]:
        lam = 0.3 if merge_method == "ta" else 0.0
        method_name = f"{merge_method.upper()}" + (f" (λ={lam})" if merge_method == "ta" else "")
        print(f"\n================= {method_name} Model Merging =================")
        
        # Uncalibrated baseline
        merged_model = merge_models(experts, base_model, method=merge_method, lam=lam)
        uncal_accs = evaluate_merged_task_specific(merged_model, heads, [], device)
        results[f"{merge_method}_uncalibrated"] = uncal_accs
        print(f"Uncalibrated: MNIST: {uncal_accs['mnist']:.2f}%, FMNIST: {uncal_accs['fmnist']:.2f}%, CIFAR10: {uncal_accs['cifar10']:.2f}%, Avg: {uncal_accs['average']:.2f}%")
        
        # Uncalibrated + Head SFT (N=64)
        print("Evaluating Uncalibrated + Head SFT (N=64)...")
        uncal_model_sft = merge_models(experts, base_model, method=merge_method, lam=lam)
        adapted_heads_uncal = run_head_sft_task_specific(uncal_model_sft, heads, datasets_list, 64, [], device)
        uncal_sft_accs = evaluate_merged_task_specific(uncal_model_sft, adapted_heads_uncal, [], device)
        results[f"{merge_method}_uncalibrated_sft_N64"] = uncal_sft_accs
        print(f"Uncalibrated + SFT (N=64): MNIST: {uncal_sft_accs['mnist']:.2f}%, FMNIST: {uncal_sft_accs['fmnist']:.2f}%, CIFAR10: {uncal_sft_accs['cifar10']:.2f}%, Avg: {uncal_sft_accs['average']:.2f}%")

        # Sweep over task-specific calibration methods and budgets N
        cal_methods = ["ts-sp-taac", "ts-tcac", "ts-l-fdsa", "ts-c-fdsa"]
        for N in [16, 64, 128]:
            print(f"\n--- Calibration Budget N = {N} ---")
            
            for cal_method in cal_methods:
                print(f"Evaluating {cal_method.upper()} (N={N})...")
                cal_model = merge_models(experts, base_model, method=merge_method, lam=lam)
                hooks, hook_objs = run_task_specific_calibration(cal_model, experts, heads, datasets_list, N, device, calibration_type=cal_method)
                accs = evaluate_merged_task_specific(cal_model, heads, hook_objs, device)
                remove_hooks(hooks)
                
                results[f"{merge_method}_{cal_method}_N{N}"] = accs
                print(f"{cal_method.upper()} (N={N}): MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}%, Avg: {accs['average']:.2f}%")
                
                # Evaluation + SFT for N=64
                if N == 64:
                    print(f"Evaluating {cal_method.upper()} + Head SFT (N=64)...")
                    cal_model_sft = merge_models(experts, base_model, method=merge_method, lam=lam)
                    hooks, hook_objs = run_task_specific_calibration(cal_model_sft, experts, heads, datasets_list, N, device, calibration_type=cal_method)
                    adapted_heads = run_head_sft_task_specific(cal_model_sft, heads, datasets_list, N, hook_objs, device)
                    sft_accs = evaluate_merged_task_specific(cal_model_sft, adapted_heads, hook_objs, device)
                    remove_hooks(hooks)
                    
                    results[f"{merge_method}_{cal_method}_sft_N64"] = sft_accs
                    print(f"{cal_method.upper()} + SFT (N=64): MNIST: {sft_accs['mnist']:.2f}%, FMNIST: {sft_accs['fmnist']:.2f}%, CIFAR10: {sft_accs['cifar10']:.2f}%, Avg: {sft_accs['average']:.2f}%")

    # Write full detailed summary of experimental results to a text file
    with open("task_specific_results.txt", "w") as f:
        f.write("Multi-Task Model Merging: Task-Specific Baselines & Methods Results Summary\n")
        f.write("========================================================================\n\n")
        f.write(f"Oracle Experts Average: {results['Oracle']['average']:.2f}%\n")
        f.write(f"MNIST Oracle: {results['Oracle']['mnist']:.2f}%\n")
        f.write(f"FMNIST Oracle: {results['Oracle']['fmnist']:.2f}%\n")
        f.write(f"CIFAR10 Oracle: {results['Oracle']['cifar10']:.2f}%\n\n")
        
        for m in ["wa", "ta"]:
            f.write(f"Merge Method: {m.upper()}\n")
            f.write(f"  Uncalibrated: {results[f'{m}_uncalibrated']['average']:.2f}%\n")
            f.write(f"  Uncalibrated + Head SFT (N=64): {results[f'{m}_uncalibrated_sft_N64']['average']:.2f}%\n")
            for N in [16, 64, 128]:
                f.write(f"  N = {N}:\n")
                for cal_method in cal_methods:
                    f.write(f"    {cal_method.upper()}: {results[f'{m}_{cal_method}_N{N}']['average']:.2f}%\n")
                    if N == 64:
                        f.write(f"    {cal_method.upper()} + Head SFT: {results[f'{m}_{cal_method}_sft_N64']['average']:.2f}%\n")
            f.write("\n")
    print("\nSaved summary of all task-specific experimental results to task_specific_results.txt")
