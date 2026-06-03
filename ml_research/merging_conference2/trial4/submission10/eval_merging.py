import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import copy
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

# 3. Evaluation on all test sets
def evaluate_merged(model, heads, device):
    datasets_list = ["mnist", "fmnist", "cifar10"]
    accuracies = {}
    
    for ds in datasets_list:
        test_dataset = get_dataset(ds, train=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
        
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

# 4. Calibration Hook Classes
class SPTAAC_Hook:
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_stds = []
        self.target_std = None
        self.merged_std = None
        self.gamma = 1.0

    def start_expert_calibration(self):
        self.calibrating_target = True
        self.expert_stds = []

    def save_expert_std(self, std):
        self.expert_stds.append(std)

    def finalize_target(self):
        self.calibrating_target = False
        self.target_std = sum(self.expert_stds) / len(self.expert_stds)

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, std):
        self.calibrating_merged = False
        self.merged_std = std
        self.gamma = self.target_std / (self.merged_std + 1e-5)
        # Prevent extreme scaling factors
        self.gamma = clip_scale(self.gamma)
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            # Output shape: (B, C, H, W)
            std = torch.sqrt(torch.var(output, dim=(0, 1, 2, 3), unbiased=False) + 1e-5).item()
            self.save_expert_std(std)
            return output
            
        if self.calibrating_merged:
            std = torch.sqrt(torch.var(output, dim=(0, 1, 2, 3), unbiased=False) + 1e-5).item()
            self.finalize_merged(std)
            return output
            
        if self.active:
            return output * self.gamma
            
        return output

class FDSA_Hook:
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_mags = []
        self.target_mag = None # Will be of shape (H, W)
        self.merged_mag = None # Will be of shape (H, W)
        self.gamma = 1.0 # Will be of shape (H, W)

    def start_expert_calibration(self):
        self.calibrating_target = True
        self.expert_mags = []

    def finalize_target(self):
        self.calibrating_target = False
        self.target_mag = torch.stack(self.expert_mags).mean(dim=0)

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, mag):
        self.calibrating_merged = False
        self.merged_mag = mag
        self.gamma = self.target_mag / (self.merged_mag + 1e-5)
        # Clamp gamma values to prevent runaway noise amplification
        self.gamma = clip_scale(self.gamma)
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            # Compute 2D FFT along spatial dimensions (-2, -1)
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            # Average across batch (dim 0) and channel (dim 1)
            mean_mag = torch.mean(mag, dim=(0, 1)) # shape (H, W)
            self.expert_mags.append(mean_mag.detach().cpu())
            return output
            
        if self.calibrating_merged:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=(0, 1)) # shape (H, W)
            self.finalize_merged(mean_mag.detach().to(output.device))
            return output
            
        if self.active:
            # Perform Frequency Domain Spectral Alignment
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            # Broadcast gamma of shape (H, W) to (B, C, H, W)
            scaled_fft = fft_out * self.gamma.view(1, 1, self.gamma.shape[0], self.gamma.shape[1])
            # Inverse FFT and take real part
            calibrated_out = torch.real(torch.fft.ifft2(scaled_fft, dim=(-2, -1)))
            return calibrated_out
            
        return output

def clip_scale(scale, max_val=5.0):
    if isinstance(scale, torch.Tensor):
        return torch.clamp(scale, 1.0/max_val, max_val)
    return min(max(scale, 1.0/max_val), max_val)

# Helper to register and handle hooks
def register_calibration_hooks(model, hook_class):
    hooks = []
    hook_objects = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hook_obj = hook_class()
            h = module.register_forward_hook(hook_obj.hook_fn)
            hooks.append(h)
            hook_objects.append(hook_obj)
    return hooks, hook_objects

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

# Calibration execution
def run_calibration(model, experts, heads, datasets_list, N, device, calibration_type="sp-taac"):
    if calibration_type == "sp-taac":
        hooks, hook_objs = register_calibration_hooks(model, SPTAAC_Hook)
    elif calibration_type == "fdsa":
        hooks, hook_objs = register_calibration_hooks(model, FDSA_Hook)
    else:
        raise ValueError(f"Unknown calibration type {calibration_type}")
        
    # Phase 1: Collect Target statistics from Experts on task-specific calibration sets
    for i, ds in enumerate(datasets_list):
        # Set up calibration dataloader of size N
        train_dataset = get_dataset(ds, train=True)
        generator = torch.Generator().manual_seed(100 + i) # different seed per task
        calib_indices = torch.randperm(len(train_dataset), generator=generator)[:N].tolist()
        calib_subset = Subset(train_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=N, shuffle=False)
        
        expert = experts[ds]
        # Attach temporary hooks to the expert to collect statistics
        if calibration_type == "sp-taac":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, SPTAAC_Hook)
        else:
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, FDSA_Hook)
            
        for h_obj in exp_hook_objs:
            h_obj.start_expert_calibration()
            
        # Single forward pass to collect stats
        inputs, _ = next(iter(calib_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = expert(inputs)
            
        # Transfer gathered stats to our main merged model hooks
        for exp_h_obj, main_h_obj in zip(exp_hook_objs, hook_objs):
            if calibration_type == "sp-taac":
                main_h_obj.expert_stds.append(exp_h_obj.expert_stds[0])
            else:
                main_h_obj.expert_mags.append(exp_h_obj.expert_mags[0])
                
        # Remove expert hooks
        remove_hooks(exp_hooks)
        
    # Finalize target statistics
    for main_h_obj in hook_objs:
        main_h_obj.finalize_target()
        
    # Phase 2: Collect Merged statistics on the Joint calibration set
    # Joint dataset size is 3 * N
    joint_inputs = []
    for i, ds in enumerate(datasets_list):
        train_dataset = get_dataset(ds, train=True)
        generator = torch.Generator().manual_seed(100 + i)
        calib_indices = torch.randperm(len(train_dataset), generator=generator)[:N].tolist()
        calib_subset = Subset(train_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=N, shuffle=False)
        inputs, _ = next(iter(calib_loader))
        joint_inputs.append(inputs)
        
    joint_inputs = torch.cat(joint_inputs, dim=0).to(device)
    
    # Enable calibration gathering on merged model
    for main_h_obj in hook_objs:
        main_h_obj.start_merged_calibration()
        
    # Forward pass of joint calibration set
    # Swap head of merged model to some dummy/first head during stats collection
    model.fc = copy.deepcopy(heads[datasets_list[0]]).to(device)
    with torch.no_grad():
        _ = model(joint_inputs)
        
    return hooks, hook_objs

# 5. Head Adaptation via Supervised SFT (from REDA)
def run_head_sft(model, heads, datasets_list, N, device, epochs=15):
    adapted_heads = {}
    model.eval()
    
    for i, ds in enumerate(datasets_list):
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
                    # Temporarily replace fc with Identity to get features
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
    
    # Evaluate Experts alone (Oracles)
    print("\nEvaluating Individual Expert Oracles...")
    oracle_accs = {}
    for ds in datasets_list:
        acc = evaluate_merged(experts[ds], heads, device)[ds]
        oracle_accs[ds] = acc
    oracle_accs["average"] = sum(oracle_accs.values()) / len(datasets_list)
    results["Oracle"] = oracle_accs
    print(f"Oracle: MNIST: {oracle_accs['mnist']:.2f}%, FMNIST: {oracle_accs['fmnist']:.2f}%, CIFAR10: {oracle_accs['cifar10']:.2f}%, Avg: {oracle_accs['average']:.2f}%")

    # Evaluate Merging Configurations
    for merge_method in ["wa", "ta"]:
        lam = 0.3 if merge_method == "ta" else 0.0
        method_name = f"{merge_method.upper()}" + (f" (λ={lam})" if merge_method == "ta" else "")
        print(f"\n================= {method_name} Model Merging =================")
        
        # 1. Uncalibrated
        merged_model = merge_models(experts, base_model, method=merge_method, lam=lam)
        uncal_accs = evaluate_merged(merged_model, heads, device)
        results[f"{merge_method}_uncalibrated"] = uncal_accs
        print(f"Uncalibrated: MNIST: {uncal_accs['mnist']:.2f}%, FMNIST: {uncal_accs['fmnist']:.2f}%, CIFAR10: {uncal_accs['cifar10']:.2f}%, Avg: {uncal_accs['average']:.2f}%")
        
        # Uncalibrated + Head SFT (N=64)
        print("Evaluating Uncalibrated + Head SFT (N=64)...")
        uncal_model_sft = merge_models(experts, base_model, method=merge_method, lam=lam)
        adapted_heads_uncal = run_head_sft(uncal_model_sft, heads, datasets_list, 64, device)
        uncal_sft_accs = evaluate_merged(uncal_model_sft, adapted_heads_uncal, device)
        results[f"{merge_method}_uncalibrated_sft_N64"] = uncal_sft_accs
        print(f"Uncalibrated + SFT (N=64): MNIST: {uncal_sft_accs['mnist']:.2f}%, FMNIST: {uncal_sft_accs['fmnist']:.2f}%, CIFAR10: {uncal_sft_accs['cifar10']:.2f}%, Avg: {uncal_sft_accs['average']:.2f}%")

        # Sweep over N for SP-TAAC and FDSA
        for N in [16, 64, 128]:
            print(f"\n--- Calibration Budget N = {N} ---")
            
            # 2. SP-TAAC Calibration
            cal_model_sptaac = merge_models(experts, base_model, method=merge_method, lam=lam)
            hooks, hook_objs = run_calibration(cal_model_sptaac, experts, heads, datasets_list, N, device, calibration_type="sp-taac")
            sptaac_accs = evaluate_merged(cal_model_sptaac, heads, device)
            remove_hooks(hooks)
            results[f"{merge_method}_sptaac_N{N}"] = sptaac_accs
            print(f"SP-TAAC (N={N}): MNIST: {sptaac_accs['mnist']:.2f}%, FMNIST: {sptaac_accs['fmnist']:.2f}%, CIFAR10: {sptaac_accs['cifar10']:.2f}%, Avg: {sptaac_accs['average']:.2f}%")
            
            # SP-TAAC + SFT (Only for N=64 as a baseline comparison)
            if N == 64:
                print("Evaluating SP-TAAC + Head SFT (N=64)...")
                cal_model_sptaac_sft = merge_models(experts, base_model, method=merge_method, lam=lam)
                hooks, hook_objs = run_calibration(cal_model_sptaac_sft, experts, heads, datasets_list, N, device, calibration_type="sp-taac")
                adapted_heads_sptaac = run_head_sft(cal_model_sptaac_sft, heads, datasets_list, N, device)
                sptaac_sft_accs = evaluate_merged(cal_model_sptaac_sft, adapted_heads_sptaac, device)
                remove_hooks(hooks)
                results[f"{merge_method}_sptaac_sft_N64"] = sptaac_sft_accs
                print(f"SP-TAAC + SFT (N=64): MNIST: {sptaac_sft_accs['mnist']:.2f}%, FMNIST: {sptaac_sft_accs['fmnist']:.2f}%, CIFAR10: {sptaac_sft_accs['cifar10']:.2f}%, Avg: {sptaac_sft_accs['average']:.2f}%")

            # 3. Frequency-Domain Spectral Alignment (FDSA) (Our Visionary Idea!)
            cal_model_fdsa = merge_models(experts, base_model, method=merge_method, lam=lam)
            hooks, hook_objs = run_calibration(cal_model_fdsa, experts, heads, datasets_list, N, device, calibration_type="fdsa")
            fdsa_accs = evaluate_merged(cal_model_fdsa, heads, device)
            remove_hooks(hooks)
            results[f"{merge_method}_fdsa_N{N}"] = fdsa_accs
            print(f"FDSA (Ours, N={N}): MNIST: {fdsa_accs['mnist']:.2f}%, FMNIST: {fdsa_accs['fmnist']:.2f}%, CIFAR10: {fdsa_accs['cifar10']:.2f}%, Avg: {fdsa_accs['average']:.2f}%")
            
            # FDSA + SFT (Only for N=64 as our proposed final joint method!)
            if N == 64:
                print("Evaluating FDSA (Ours) + Head SFT (N=64)...")
                cal_model_fdsa_sft = merge_models(experts, base_model, method=merge_method, lam=lam)
                hooks, hook_objs = run_calibration(cal_model_fdsa_sft, experts, heads, datasets_list, N, device, calibration_type="fdsa")
                adapted_heads_fdsa = run_head_sft(cal_model_fdsa_sft, heads, datasets_list, N, device)
                fdsa_sft_accs = evaluate_merged(cal_model_fdsa_sft, adapted_heads_fdsa, device)
                remove_hooks(hooks)
                results[f"{merge_method}_fdsa_sft_N64"] = fdsa_sft_accs
                print(f"FDSA + SFT (Ours, N=64): MNIST: {fdsa_sft_accs['mnist']:.2f}%, FMNIST: {fdsa_sft_accs['fmnist']:.2f}%, CIFAR10: {fdsa_sft_accs['cifar10']:.2f}%, Avg: {fdsa_sft_accs['average']:.2f}%")

    # Write summary of results to a text file
    with open("experimental_results.txt", "w") as f:
        f.write("Multi-Task Model Merging Experimental Results Summary\n")
        f.write("====================================================\n\n")
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
                f.write(f"    SP-TAAC: {results[f'{m}_sptaac_N{N}']['average']:.2f}%\n")
                if N == 64:
                    f.write(f"    SP-TAAC + Head SFT: {results[f'{m}_sptaac_sft_N64']['average']:.2f}%\n")
                f.write(f"    FDSA (Ours): {results[f'{m}_fdsa_N{N}']['average']:.2f}%\n")
                if N == 64:
                    f.write(f"    FDSA (Ours) + Head SFT: {results[f'{m}_fdsa_sft_N64']['average']:.2f}%\n")
            f.write("\n")
    print("\nSaved summary of experimental results to experimental_results.txt")
