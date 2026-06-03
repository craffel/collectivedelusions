import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import os
import copy
import json
import numpy as np

# Set device and disable cuDNN if it causes issues
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors.")

# -------------------------------------------------------------
# 1. Dataset Preparation
# -------------------------------------------------------------
print("Preparing datasets...")
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # replicate to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_mnist = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_gray)
test_mnist = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_gray)

train_fmnist = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_gray)
test_fmnist = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_gray)

train_cifar = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_rgb)
test_cifar = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_rgb)

# Create DataLoaders
batch_size = 256
loader_args = {"batch_size": batch_size, "num_workers": 2, "pin_memory": True}

train_loaders = {
    "mnist": DataLoader(train_mnist, shuffle=True, **loader_args),
    "fmnist": DataLoader(train_fmnist, shuffle=True, **loader_args),
    "cifar10": DataLoader(train_cifar, shuffle=True, **loader_args)
}

test_loaders = {
    "mnist": DataLoader(test_mnist, shuffle=False, **loader_args),
    "fmnist": DataLoader(test_fmnist, shuffle=False, **loader_args),
    "cifar10": DataLoader(test_cifar, shuffle=False, **loader_args)
}

# -------------------------------------------------------------
# 2. Expert Training or Loading
# -------------------------------------------------------------
experts_dir = "./experts"
os.makedirs(experts_dir, exist_ok=True)

print("Loading ImageNet-pretrained ResNet-18 progenitor...")
progenitor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
progenitor_state = copy.deepcopy(progenitor.state_dict())

def get_expert(task_name):
    expert_path = os.path.join(experts_dir, f"expert_{task_name}.pt")
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(expert_path, map_location=device))
    return model

# Load pre-trained experts
print("Loading pre-trained experts...")
expert_mnist = get_expert("mnist")
expert_fmnist = get_expert("fmnist")
expert_cifar10 = get_expert("cifar10")

experts = {
    "mnist": expert_mnist,
    "fmnist": expert_fmnist,
    "cifar10": expert_cifar10
}

# -------------------------------------------------------------
# 3. Evaluation Function
# -------------------------------------------------------------
def evaluate_model(backbone, experts, test_loaders):
    backbone.eval()
    results = {}
    with torch.no_grad():
        for task_name, loader in test_loaders.items():
            eval_model = copy.deepcopy(backbone)
            eval_model.fc = copy.deepcopy(experts[task_name].fc)
            eval_model = eval_model.to(device)
            eval_model.eval()
            
            correct = 0
            total = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = eval_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            accuracy = 100.0 * correct / total
            results[task_name] = accuracy
            print(f"Task: {task_name:10s} | Accuracy: {accuracy:.2f}%")
            
    avg_accuracy = np.mean(list(results.values()))
    results["average"] = avg_accuracy
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    return results

print("\n--- Evaluating Oracle Experts (No Merging) ---")
oracle_results = {}
for task_name in ["mnist", "fmnist", "cifar10"]:
    oracle_results[task_name] = evaluate_model(experts[task_name], {task_name: experts[task_name]}, {task_name: test_loaders[task_name]})[task_name]
oracle_results["average"] = np.mean(list(oracle_results.values()))
print(f"Oracle Experts Average: {oracle_results['average']:.2f}%")

# -------------------------------------------------------------
# 4. Standard Merging Functions
# -------------------------------------------------------------
def get_standard_merge(models_dict, progenitor_state, merge_type="WA", lam=0.2):
    keys = [k for k in progenitor_state.keys() if "fc" not in k]
    merged_state = {}
    
    if merge_type == "WA":
        for key in keys:
            tensors = [m.state_dict()[key] for m in models_dict.values()]
            if torch.is_floating_point(tensors[0]):
                merged_state[key] = torch.mean(torch.stack(tensors), dim=0)
            else:
                merged_state[key] = tensors[0]
    elif merge_type == "TA":
        for key in keys:
            if "running_mean" in key or "running_var" in key:
                # BatchNorm running statistics must be averaged, not arithmetically manipulated
                tensors = [m.state_dict()[key] for m in models_dict.values()]
                merged_state[key] = torch.mean(torch.stack(tensors), dim=0)
            elif torch.is_floating_point(progenitor_state[key]):
                task_vectors = [m.state_dict()[key].cpu() - progenitor_state[key].cpu() for m in models_dict.values()]
                merged_state[key] = progenitor_state[key].cpu() + lam * torch.sum(torch.stack(task_vectors), dim=0)
            else:
                merged_state[key] = progenitor_state[key].cpu()
            
    merged_model = models.resnet18()
    merged_model.fc = nn.Linear(512, 10)
    merged_model.load_state_dict(merged_state, strict=False)
    return merged_model

# -------------------------------------------------------------
# 5. Calibration Method Implementations
# -------------------------------------------------------------

# --- Corrected SP-TAAC Baseline (Real-data Calibration) ---
def apply_sp_taac(merged_model, experts_dict, train_loaders, num_samples=128):
    """
    Corrected SP-TAAC calibration. Runs each expert ONLY on its own task's calibration data.
    """
    print(f"Applying Corrected SP-TAAC baseline with N={num_samples} samples per task...")
    merged_model = copy.deepcopy(merged_model).to(device)
    merged_model.eval()
    
    # Create task-specific calibration inputs
    task_calib = {}
    for task_name, loader in train_loaders.items():
        calib_inputs = []
        count = 0
        for inputs, _ in loader:
            calib_inputs.append(inputs)
            count += inputs.size(0)
            if count >= num_samples:
                break
        task_calib[task_name] = torch.cat(calib_inputs, dim=0)[:num_samples].to(device)
        
    # Get all BN layers in execution order
    conv_bn_pairs = []
    modules_dict = dict(merged_model.named_modules())
    for name, module in merged_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            conv_bn_pairs.append(name)
            
    # Registers hooks for the inputs to all BatchNorm2d layers
    merged_container = {bn_name: [] for bn_name in conv_bn_pairs}
    expert_containers = {task: {bn_name: [] for bn_name in conv_bn_pairs} for task in experts_dict.keys()}
    
    # Hooks
    merged_hooks = []
    for bn_name in conv_bn_pairs:
        m_bn = modules_dict[bn_name]
        h = m_bn.register_forward_hook(lambda mod, inp, out, name=bn_name: merged_container[name].append(inp[0].detach().cpu()))
        merged_hooks.append(h)
        
    expert_hooks = {task: [] for task in experts_dict.keys()}
    for task, exp_model in experts_dict.items():
        exp_model = exp_model.to(device)
        exp_model.eval()
        exp_modules = dict(exp_model.named_modules())
        for bn_name in conv_bn_pairs:
            e_bn = exp_modules[bn_name]
            h = e_bn.register_forward_hook(lambda mod, inp, out, t=task, name=bn_name: expert_containers[t][name].append(inp[0].detach().cpu()))
            expert_hooks[task].append(h)
            
    # Run forward passes on calibration data
    with torch.no_grad():
        # Run merged model on all calibration data
        for task, cal_data in task_calib.items():
            _ = merged_model(cal_data)
            
        # Run each expert ONLY on its own task's calibration data
        for task, exp_model in experts_dict.items():
            _ = exp_model(task_calib[task])
            
    # Remove hooks
    for h in merged_hooks:
        h.remove()
    for task in experts_dict.keys():
        for h in expert_hooks[task]:
            h.remove()
            
    # Now, compute scale correction factors and update BN parameters
    with torch.no_grad():
        for bn_name in conv_bn_pairs:
            m_bn = modules_dict[bn_name]
            
            # Concatenate collected activations
            v_merged = torch.cat(merged_container[bn_name], dim=0) # (3 * N_samples, C, H, W)
            
            # Concatenate expert activations over their respective tasks
            v_experts = []
            for task in experts_dict.keys():
                v_experts.append(torch.cat(expert_containers[task][bn_name], dim=0))
            v_target = torch.cat(v_experts, dim=0) # (3 * N_samples, C, H, W)
            
            # Compute channel-wise standard deviations
            std_merged = torch.std(v_merged, dim=(0, 2, 3))
            std_target = torch.std(v_target, dim=(0, 2, 3))
            
            # Scale factor
            gamma = std_target / (std_merged + 1e-8)
            gamma = torch.clamp(gamma, min=0.1, max=10.0).to(device)
            
            # Scale BN weights and biases in-place
            m_bn.weight.copy_(gamma * m_bn.weight)
            if m_bn.bias is not None:
                m_bn.bias.copy_(gamma * m_bn.bias)
                
    print("SP-TAAC calibration completed.")
    return merged_model


# --- Proposed Weight-Level IPR (W-IPR) ---
def apply_weight_level_ipr(merged_model, experts_dict):
    print("Applying proposed Weight-Level Isotropic Parameter Resonance (W-IPR)...")
    merged_model = copy.deepcopy(merged_model).to(device)
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                norm_merged = torch.norm(module.weight, p="fro")
                norms_experts = torch.tensor([torch.norm(exp_mod.weight, p="fro") for exp_mod in expert_modules])
                avg_norm_experts = torch.mean(norms_experts)
                R = norm_merged / (avg_norm_experts + 1e-8)
                R = torch.clamp(R, min=0.1, max=10.0)
                module.weight.copy_(module.weight / R)
                if module.bias is not None:
                    module.bias.copy_(module.bias / R)
    print("W-IPR completed.")
    return merged_model


# --- Proposed BN-Level IPR (BN-IPR) ---
def apply_bn_level_ipr(merged_model, experts_dict):
    print("Applying proposed BN-Level Isotropic Parameter Resonance (BN-IPR)...")
    merged_model = copy.deepcopy(merged_model).to(device)
    resonance_ratios = {}
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                norm_merged = torch.norm(module.weight, p="fro")
                norms_experts = torch.tensor([torch.norm(exp_mod.weight, p="fro") for exp_mod in expert_modules])
                avg_norm_experts = torch.mean(norms_experts)
                R = norm_merged / (avg_norm_experts + 1e-8)
                R = torch.clamp(R, min=0.1, max=10.0)
                resonance_ratios[name] = R.item()
                
        last_conv_r = 1.0
        last_conv_name = None
        for name, module in merged_model.named_modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                last_conv_name = name
                last_conv_r = resonance_ratios[name]
            elif isinstance(module, nn.BatchNorm2d):
                if last_conv_name is not None:
                    module.running_mean.copy_(last_conv_r * module.running_mean)
                    module.running_var.copy_((last_conv_r ** 2) * module.running_var)
    print("BN-IPR completed.")
    return merged_model


# --- Proposed Update-Level IPR (U-IPR) ---
def apply_update_level_ipr(merged_model, experts_dict, progenitor_state, scale_bn=True):
    """
    Proposed Update-level Isotropic Parameter Resonance (U-IPR).
    Rescales the task vectors of Conv/Linear layers (and optionally BatchNorm weights) to preserve update norms.
    """
    print("Applying proposed Update-Level Isotropic Parameter Resonance (U-IPR)...")
    merged_model = copy.deepcopy(merged_model).to(device)
    
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            # Process Conv2d and Linear layers (excluding fc head)
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # Get progenitor weight
                w_init = progenitor_state[f"{name}.weight"].to(device)
                
                # Compute expert updates (task vectors)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = module.weight - w_init
                
                # Compute Frobenius norms of updates
                norm_t_merged = torch.norm(t_merged, p="fro")
                norms_t_experts = torch.tensor([torch.norm(t_exp, p="fro") for t_exp in t_experts])
                avg_norm_t_experts = torch.mean(norms_t_experts)
                
                # Update resonance ratio
                S = avg_norm_t_experts / (norm_t_merged + 1e-8)
                S = torch.clamp(S, min=0.1, max=10.0) # We scale updates to match optimal expert norms
                
                # Update weight: W_new = W_init + S * T_merged
                new_weight = w_init + S * t_merged
                module.weight.copy_(new_weight)
                
                # Process bias if it exists
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    new_bias = b_init + S * b_merged
                    module.bias.copy_(new_bias)
                    
                print(f"U-IPR Layer: {name:30s} | Merged Update Norm: {norm_t_merged.item():8.5f} | Avg Expert Update Norm: {avg_norm_t_experts.item():8.5f} | S: {S.item():.4f}")
                
            # Process BatchNorm layers (scale BN weights/biases as updates)
            elif isinstance(module, nn.BatchNorm2d) and scale_bn:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # For BN weights (gamma)
                w_init = progenitor_state[f"{name}.weight"].to(device)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = module.weight - w_init
                norm_t_merged = torch.norm(t_merged, p="fro")
                norms_t_experts = torch.tensor([torch.norm(t_exp, p="fro") for t_exp in t_experts])
                avg_norm_t_experts = torch.mean(norms_t_experts)
                
                S_w = avg_norm_t_experts / (norm_t_merged + 1e-8)
                S_w = torch.clamp(S_w, min=0.1, max=10.0)
                module.weight.copy_(w_init + S_w * t_merged)
                
                # For BN biases (beta)
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    norm_b_merged = torch.norm(b_merged, p="fro")
                    norms_b_experts = torch.tensor([torch.norm(b_exp, p="fro") for b_exp in b_experts])
                    avg_norm_b_experts = torch.mean(norms_b_experts)
                    
                    S_b = avg_norm_b_experts / (norm_b_merged + 1e-8)
                    S_b = torch.clamp(S_b, min=0.1, max=10.0)
                    module.bias.copy_(b_init + S_b * b_merged)
                    
    print("U-IPR completed.")
    return merged_model


# --- Proposed Spectral Parameter Resonance (S-IPR) ---
def apply_spectral_parameter_resonance(merged_model, experts_dict, progenitor_state, scale_bn=True):
    """
    Proposed Spectral Parameter Resonance (S-IPR).
    Performs SVD on the update (task vector) of Conv/Linear layers and directly aligns the singular value spectrum
    to the average singular value spectrum of the individual experts. For BatchNorm layers and biases, it falls
    back to U-IPR's magnitude scaling.
    """
    print("Applying proposed Spectral Parameter Resonance (S-IPR)...")
    merged_model = copy.deepcopy(merged_model).to(device)
    
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            # Process Conv2d and Linear layers (excluding fc head)
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # Get progenitor weight
                w_init = progenitor_state[f"{name}.weight"].to(device)
                
                # Compute expert updates (task vectors)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = module.weight - w_init
                
                # Reshape updates to 2D for SVD
                orig_shape = t_merged.shape
                t_experts_2d = [t.view(t.shape[0], -1) for t in t_experts]
                t_merged_2d = t_merged.view(t_merged.shape[0], -1)
                
                # SVD of expert updates
                s_experts = []
                for t_exp_2d in t_experts_2d:
                    _, S_exp, _ = torch.linalg.svd(t_exp_2d, full_matrices=False)
                    s_experts.append(S_exp)
                # Average singular value spectrum of experts
                avg_S_experts = torch.mean(torch.stack(s_experts), dim=0)
                
                # SVD of merged update
                U_merged, S_merged, Vh_merged = torch.linalg.svd(t_merged_2d, full_matrices=False)
                
                # Directly reconstruct updated 2D matrix using the average singular value spectrum of experts
                t_corrected_2d = U_merged @ torch.diag(avg_S_experts) @ Vh_merged
                
                # Update weight: W_new = W_init + T_corrected
                new_weight = w_init + t_corrected_2d.view(orig_shape)
                module.weight.copy_(new_weight)
                
                # Process bias if it exists (using U-IPR magnitude scaling as fallback)
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    norm_b_merged = torch.norm(b_merged, p="fro")
                    norms_b_experts = torch.tensor([torch.norm(b_exp, p="fro") for b_exp in b_experts])
                    avg_norm_b_experts = torch.mean(norms_b_experts)
                    S_b = avg_norm_b_experts / (norm_b_merged + 1e-8)
                    S_b = torch.clamp(S_b, min=0.1, max=10.0)
                    new_bias = b_init + S_b * b_merged
                    module.bias.copy_(new_bias)
                    
                print(f"S-IPR Layer: {name:30s} | Singular Values Count: {len(avg_S_experts):4d} | Max Avg S: {avg_S_experts[0].item():8.5f} | Max Merged S: {S_merged[0].item():8.5f}")
                
            # Process BatchNorm layers (scale BN weights/biases as updates, falling back to U-IPR)
            elif isinstance(module, nn.BatchNorm2d) and scale_bn:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # For BN weights (gamma)
                w_init = progenitor_state[f"{name}.weight"].to(device)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = module.weight - w_init
                norm_t_merged = torch.norm(t_merged, p="fro")
                norms_t_experts = torch.tensor([torch.norm(t_exp, p="fro") for t_exp in t_experts])
                avg_norm_t_experts = torch.mean(norms_t_experts)
                
                S_w = avg_norm_t_experts / (norm_t_merged + 1e-8)
                S_w = torch.clamp(S_w, min=0.1, max=10.0)
                module.weight.copy_(w_init + S_w * t_merged)
                
                # For BN biases (beta)
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    norm_b_merged = torch.norm(b_merged, p="fro")
                    norms_b_experts = torch.tensor([torch.norm(b_exp, p="fro") for b_exp in b_experts])
                    avg_norm_b_experts = torch.mean(norms_b_experts)
                    
                    S_b = avg_norm_b_experts / (norm_b_merged + 1e-8)
                    S_b = torch.clamp(S_b, min=0.1, max=10.0)
                    module.bias.copy_(b_init + S_b * b_merged)
                    
    print("S-IPR completed.")
    return merged_model


# --- Proposed Subspace-Aligned Isotropic Parameter Resonance (SA-IPR) ---
def apply_subspace_aligned_ipr(merged_model, experts_dict, progenitor_state, alpha=0.5, scale_bn=True):
    """
    Proposed Subspace-Aligned Isotropic Parameter Resonance (SA-IPR).
    Projects expert task updates onto their leading joint singular subspace to filter out destructive orthogonal
    interference, averages the aligned updates, and scales them back to the target average norm of the experts (U-IPR).
    For BatchNorm layers and biases, it falls back to U-IPR's magnitude scaling.
    """
    print(f"Applying proposed Subspace-Aligned Isotropic Parameter Resonance (SA-IPR) with alpha={alpha}...")
    merged_model = copy.deepcopy(merged_model).to(device)
    
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            # Process Conv2d and Linear layers (excluding fc head)
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # Get progenitor weight
                w_init = progenitor_state[f"{name}.weight"].to(device)
                
                # Compute expert updates (task vectors)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                
                # Reshape updates to 2D for SVD
                orig_shape = t_experts[0].shape
                t_experts_2d = [t.view(t.shape[0], -1) for t in t_experts]
                
                R, C = t_experts_2d[0].shape
                k_sing = min(R, C)
                # Compute rank r to keep based on alpha
                r = max(1, int(alpha * k_sing))
                
                # Compute SVD for each expert to get leading singular vectors
                U_list = []
                V_list = []
                for t_exp_2d in t_experts_2d:
                    U_exp, S_exp, Vh_exp = torch.linalg.svd(t_exp_2d, full_matrices=False)
                    U_list.append(U_exp[:, :r])
                    V_list.append(Vh_exp[:r, :].T)  # V is columns of Vh transposed
                
                # Concatenate leading singular vectors
                U_concat = torch.cat(U_list, dim=1) # Shape: (R, K * r)
                V_concat = torch.cat(V_list, dim=1) # Shape: (C, K * r)
                
                # Find orthonormal basis for the joint subspaces using SVD
                U_shared, _, _ = torch.linalg.svd(U_concat, full_matrices=False)
                V_shared, _, _ = torch.linalg.svd(V_concat, full_matrices=False)
                
                # Keep top d dimensions for projection (d = r or capped at min(R, C))
                d = min(k_sing, r)
                P_U = U_shared[:, :d] # Shape: (R, d)
                P_V = V_shared[:, :d] # Shape: (C, d)
                
                # Project each expert task update onto the shared subspace
                # P_U @ P_U^T is the orthogonal projection matrix
                # T_aligned_2d = P_U @ (P_U^T @ T_2d @ P_V) @ P_V^T
                t_experts_aligned_2d = []
                for t_exp_2d in t_experts_2d:
                    t_aligned = P_U @ (P_U.T @ t_exp_2d @ P_V) @ P_V.T
                    t_experts_aligned_2d.append(t_aligned)
                
                # Average aligned expert updates
                t_merged_aligned_2d = torch.mean(torch.stack(t_experts_aligned_2d), dim=0)
                
                # Apply U-IPR scaling: target average expert norm vs merged aligned norm
                norms_t_experts = torch.tensor([torch.norm(t_exp, p="fro") for t_exp in t_experts])
                avg_norm_t_experts = torch.mean(norms_t_experts)
                
                norm_t_merged_aligned = torch.norm(t_merged_aligned_2d, p="fro")
                
                S = avg_norm_t_experts / (norm_t_merged_aligned + 1e-8)
                S = torch.clamp(S, min=0.1, max=10.0)
                
                # Reconstruct weight in original shape
                t_final = S * t_merged_aligned_2d
                new_weight = w_init + t_final.view(orig_shape)
                module.weight.copy_(new_weight)
                
                # Process bias if it exists (using U-IPR magnitude scaling as fallback)
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    norm_b_merged = torch.norm(b_merged, p="fro")
                    norms_b_experts = torch.tensor([torch.norm(b_exp, p="fro") for b_exp in b_experts])
                    avg_norm_b_experts = torch.mean(norms_b_experts)
                    S_b = avg_norm_b_experts / (norm_b_merged + 1e-8)
                    S_b = torch.clamp(S_b, min=0.1, max=10.0)
                    new_bias = b_init + S_b * b_merged
                    module.bias.copy_(new_bias)
                    
                print(f"SA-IPR Layer: {name:30s} | Rank r: {r:4d} | Proj Dim d: {d:4d} | S: {S.item():.4f}")
                
            # Process BatchNorm layers (scale BN weights/biases as updates, falling back to U-IPR)
            elif isinstance(module, nn.BatchNorm2d) and scale_bn:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # For BN weights (gamma)
                w_init = progenitor_state[f"{name}.weight"].to(device)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = module.weight - w_init
                norm_t_merged = torch.norm(t_merged, p="fro")
                norms_t_experts = torch.tensor([torch.norm(t_exp, p="fro") for t_exp in t_experts])
                avg_norm_t_experts = torch.mean(norms_t_experts)
                
                S_w = avg_norm_t_experts / (norm_t_merged + 1e-8)
                S_w = torch.clamp(S_w, min=0.1, max=10.0)
                module.weight.copy_(w_init + S_w * t_merged)
                
                # For BN biases (beta)
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    norm_b_merged = torch.norm(b_merged, p="fro")
                    norms_b_experts = torch.tensor([torch.norm(b_exp, p="fro") for b_exp in b_experts])
                    avg_norm_b_experts = torch.mean(norms_b_experts)
                    
                    S_b = avg_norm_b_experts / (norm_b_merged + 1e-8)
                    S_b = torch.clamp(S_b, min=0.1, max=10.0)
                    module.bias.copy_(b_init + S_b * b_merged)
                    
    print("SA-IPR completed.")
    return merged_model


# --- Proposed Saliency-Weighted Isotropic Parameter Resonance (I-IPR) ---
def apply_saliency_weighted_ipr(merged_model, experts_dict, progenitor_state, scale_bn=True):
    """
    Proposed Saliency-Weighted Isotropic Parameter Resonance (I-IPR).
    Merges task vectors using their squared magnitudes (saliency) as weights, then rescales
    the merged task vectors using U-IPR to preserve optimal update norms.
    """
    print("Applying proposed Saliency-Weighted Isotropic Parameter Resonance (I-IPR)...")
    merged_model = copy.deepcopy(merged_model).to(device)
    
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            # Process Conv2d and Linear layers (excluding fc head)
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # Get progenitor weight
                w_init = progenitor_state[f"{name}.weight"].to(device)
                
                # Compute expert updates (task vectors)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                
                # Compute saliency weights: (T_k)^2
                saliencies = [t_exp ** 2 for t_exp in t_experts]
                sum_saliencies = torch.stack(saliencies).sum(dim=0) + 1e-8
                
                # Saliency-weighted merge
                t_merged_saliency = torch.stack([sal * t_exp for sal, t_exp in zip(saliencies, t_experts)]).sum(dim=0) / sum_saliencies
                
                # Compute Frobenius norms of updates
                norms_t_experts = torch.tensor([torch.norm(t_exp, p="fro") for t_exp in t_experts])
                avg_norm_t_experts = torch.mean(norms_t_experts)
                
                norm_t_merged_saliency = torch.norm(t_merged_saliency, p="fro")
                
                # Update resonance ratio
                S = avg_norm_t_experts / (norm_t_merged_saliency + 1e-8)
                S = torch.clamp(S, min=0.1, max=10.0)
                
                # Update weight: W_new = W_init + S * T_merged_saliency
                new_weight = w_init + S * t_merged_saliency
                module.weight.copy_(new_weight)
                
                # Process bias if it exists (using standard U-IPR scale fallback)
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = torch.stack(b_experts).mean(dim=0)
                    norm_b_merged = torch.norm(b_merged, p="fro")
                    norms_b_experts = torch.tensor([torch.norm(b_exp, p="fro") for b_exp in b_experts])
                    avg_norm_b_experts = torch.mean(norms_b_experts)
                    S_b = avg_norm_b_experts / (norm_b_merged + 1e-8)
                    S_b = torch.clamp(S_b, min=0.1, max=10.0)
                    new_bias = b_init + S_b * b_merged
                    module.bias.copy_(new_bias)
                    
                print(f"I-IPR Layer: {name:30s} | Merged Saliency Update Norm: {norm_t_merged_saliency.item():8.5f} | Avg Expert Update Norm: {avg_norm_t_experts.item():8.5f} | S: {S.item():.4f}")
                
            # Process BatchNorm layers (scale BN weights/biases as updates, falling back to U-IPR)
            elif isinstance(module, nn.BatchNorm2d) and scale_bn:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # For BN weights (gamma)
                w_init = progenitor_state[f"{name}.weight"].to(device)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = torch.stack(t_experts).mean(dim=0)
                norm_t_merged = torch.norm(t_merged, p="fro")
                norms_t_experts = torch.tensor([torch.norm(t_exp, p="fro") for t_exp in t_experts])
                avg_norm_t_experts = torch.mean(norms_t_experts)
                
                S_w = avg_norm_t_experts / (norm_t_merged + 1e-8)
                S_w = torch.clamp(S_w, min=0.1, max=10.0)
                module.weight.copy_(w_init + S_w * t_merged)
                
                # For BN biases (beta)
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = torch.stack(b_experts).mean(dim=0)
                    norm_b_merged = torch.norm(b_merged, p="fro")
                    norms_b_experts = torch.tensor([torch.norm(b_exp, p="fro") for b_exp in b_experts])
                    avg_norm_b_experts = torch.mean(norms_b_experts)
                    
                    S_b = avg_norm_b_experts / (norm_b_merged + 1e-8)
                    S_b = torch.clamp(S_b, min=0.1, max=10.0)
                    module.bias.copy_(b_init + S_b * b_merged)
                    
    print("I-IPR completed.")
    return merged_model


# --- Proposed Orthogonality-Aware Isotropic Parameter Resonance (OA-IPR) ---
def apply_orthogonality_aware_ipr(merged_model, experts_dict, progenitor_state, scale_bn=True):
    """
    Proposed Orthogonality-Aware Isotropic Parameter Resonance (OA-IPR).
    Computes a priori resonance scaling factor directly from pairwise cosine similarities of experts' updates.
    """
    print("Applying proposed Orthogonality-Aware Isotropic Parameter Resonance (OA-IPR)...")
    merged_model = copy.deepcopy(merged_model).to(device)
    K = len(experts_dict)
    
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            # Process Conv2d and Linear layers (excluding fc head)
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # Get progenitor weight
                w_init = progenitor_state[f"{name}.weight"].to(device)
                
                # Compute expert updates (task vectors)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = module.weight - w_init
                
                # Compute pairwise cosine similarities of task updates
                pairwise_sims = []
                for i in range(K):
                    for j in range(i + 1, K):
                        # Cosine similarity
                        dot_prod = torch.sum(t_experts[i] * t_experts[j])
                        norm_i = torch.norm(t_experts[i])
                        norm_j = torch.norm(t_experts[j])
                        sim = dot_prod / (norm_i * norm_j + 1e-8)
                        pairwise_sims.append(sim)
                
                avg_sim = torch.mean(torch.stack(pairwise_sims)) if len(pairwise_sims) > 0 else torch.tensor(0.0).to(device)
                
                # Compute automatic lambda (scale factor) for this layer
                sum_t_experts = torch.stack(t_experts).sum(dim=0)
                norm_sum_t = torch.norm(sum_t_experts, p="fro")
                norm_t_merged = torch.norm(t_merged, p="fro")
                lam_l = norm_t_merged / (norm_sum_t + 1e-8)
                
                # Compute theoretical scaling factor: S_theo = 1 / (lam * sqrt(K) * sqrt(1 + (2/K) * sum(rho_ij)))
                sum_rho = torch.sum(torch.stack(pairwise_sims)) if len(pairwise_sims) > 0 else torch.tensor(0.0).to(device)
                denom = lam_l * torch.sqrt(torch.tensor(float(K)).to(device)) * torch.sqrt(1.0 + (2.0 / K) * sum_rho)
                S_theo = 1.0 / (denom + 1e-8)
                S_theo = torch.clamp(S_theo, min=0.1, max=10.0)
                
                # Update weight using S_theo
                new_weight = w_init + S_theo * t_merged
                module.weight.copy_(new_weight)
                
                # Process bias if it exists
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    new_bias = b_init + S_theo * b_merged
                    module.bias.copy_(new_bias)
                    
                print(f"OA-IPR Layer: {name:30s} | Avg Cos Sim: {avg_sim.item():8.5f} | S_theo: {S_theo.item():.4f}")
                
            # Process BatchNorm layers (scale BN weights/biases as updates, falling back to U-IPR)
            elif isinstance(module, nn.BatchNorm2d) and scale_bn:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # For BN weights (gamma)
                w_init = progenitor_state[f"{name}.weight"].to(device)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = module.weight - w_init
                norm_t_merged = torch.norm(t_merged, p="fro")
                norms_t_experts = torch.tensor([torch.norm(t_exp, p="fro") for t_exp in t_experts])
                avg_norm_t_experts = torch.mean(norms_t_experts)
                
                S_w = avg_norm_t_experts / (norm_t_merged + 1e-8)
                S_w = torch.clamp(S_w, min=0.1, max=10.0)
                module.weight.copy_(w_init + S_w * t_merged)
                
                # For BN biases (beta)
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    norm_b_merged = torch.norm(b_merged, p="fro")
                    norms_b_experts = torch.tensor([torch.norm(b_exp, p="fro") for b_exp in b_experts])
                    avg_norm_b_experts = torch.mean(norms_b_experts)
                    
                    S_b = avg_norm_b_experts / (norm_b_merged + 1e-8)
                    S_b = torch.clamp(S_b, min=0.1, max=10.0)
                    module.bias.copy_(b_init + S_b * b_merged)
                    
    print("OA-IPR completed.")
    return merged_model


# --- Proposed Channel-Level Update IPR (CU-IPR) ---
def apply_channel_update_level_ipr(merged_model, experts_dict, progenitor_state, scale_bn=True):
    """
    Proposed Channel-level Update Isotropic Parameter Resonance (CU-IPR).
    Rescales the task vectors of Conv/Linear layers (and optionally BatchNorm) channel-by-channel
    to preserve filter-level update norms.
    """
    print("Applying proposed Channel-Level Update Isotropic Parameter Resonance (CU-IPR)...")
    merged_model = copy.deepcopy(merged_model).to(device)
    
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            # Process Conv2d and Linear layers (excluding fc head)
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # Get progenitor weight
                w_init = progenitor_state[f"{name}.weight"].to(device)
                
                # Weight has shape (C_out, C_in, ...)
                C_out = module.weight.shape[0]
                
                # We will construct the new weight channel-by-channel
                new_weight = torch.zeros_like(module.weight)
                
                # We'll compute the average scaling factor for logging
                scale_factors = []
                
                for c in range(C_out):
                    w_init_c = w_init[c]
                    t_experts_c = [exp_mod.weight[c] - w_init_c for exp_mod in expert_modules]
                    t_merged_c = module.weight[c] - w_init_c
                    
                    norm_t_merged_c = torch.norm(t_merged_c, p="fro")
                    norms_t_experts_c = torch.tensor([torch.norm(t_exp, p="fro") for t_exp in t_experts_c])
                    avg_norm_t_experts_c = torch.mean(norms_t_experts_c)
                    
                    S_c = avg_norm_t_experts_c / (norm_t_merged_c + 1e-8)
                    S_c = torch.clamp(S_c, min=0.1, max=10.0)
                    scale_factors.append(S_c)
                    
                    new_weight[c] = w_init_c + S_c * t_merged_c
                    
                module.weight.copy_(new_weight)
                
                # Process bias if it exists
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    
                    new_bias = torch.zeros_like(module.bias)
                    for c in range(C_out):
                        S_c = scale_factors[c]
                        new_bias[c] = b_init[c] + S_c * b_merged[c]
                    module.bias.copy_(new_bias)
                    
                avg_S = torch.mean(torch.stack(scale_factors))
                print(f"CU-IPR Layer: {name:30s} | C_out: {C_out:4d} | Avg S_c: {avg_S.item():.4f}")
                
            # Process BatchNorm layers (scale BN weights/biases channel-wise)
            elif isinstance(module, nn.BatchNorm2d) and scale_bn:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # BN weight has shape (num_features,)
                num_features = module.weight.shape[0]
                
                # Gamma weight
                w_init = progenitor_state[f"{name}.weight"].to(device)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = module.weight - w_init
                
                new_weight = torch.zeros_like(module.weight)
                scale_factors_w = []
                for c in range(num_features):
                    t_experts_c = [t_exp[c] for t_exp in t_experts]
                    t_merged_c = t_merged[c]
                    
                    norm_t_merged_c = torch.abs(t_merged_c)
                    norms_t_experts_c = torch.tensor([torch.abs(t_exp) for t_exp in t_experts_c])
                    avg_norm_t_experts_c = torch.mean(norms_t_experts_c)
                    
                    S_c = avg_norm_t_experts_c / (norm_t_merged_c + 1e-8)
                    S_c = torch.clamp(S_c, min=0.1, max=10.0)
                    scale_factors_w.append(S_c)
                    new_weight[c] = w_init[c] + S_c * t_merged_c
                module.weight.copy_(new_weight)
                
                # Beta bias
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    
                    new_bias = torch.zeros_like(module.bias)
                    for c in range(num_features):
                        b_experts_c = [b_exp[c] for b_exp in b_experts]
                        b_merged_c = b_merged[c]
                        
                        norm_b_merged_c = torch.abs(b_merged_c)
                        norms_b_experts_c = torch.tensor([torch.abs(b_exp) for b_exp in b_experts_c])
                        avg_norm_b_experts_c = torch.mean(norms_b_experts_c)
                        
                        S_c = avg_norm_b_experts_c / (norm_b_merged_c + 1e-8)
                        S_c = torch.clamp(S_c, min=0.1, max=10.0)
                        new_bias[c] = b_init[c] + S_c * b_merged_c
                    module.bias.copy_(new_bias)
                    
    print("CU-IPR completed.")
    return merged_model


# --- Proposed Channel-Level Orthogonality-Aware IPR (CO-IPR) ---
def apply_channel_orthogonality_aware_ipr(merged_model, experts_dict, progenitor_state, scale_bn=True):
    """
    Proposed Channel-level Orthogonality-Aware Isotropic Parameter Resonance (CO-IPR).
    Computes fine-grained scaling factors channel-by-channel based on local pairwise filter cosine similarities.
    """
    print("Applying proposed Channel-Level Orthogonality-Aware Isotropic Parameter Resonance (CO-IPR)...")
    merged_model = copy.deepcopy(merged_model).to(device)
    K = len(experts_dict)
    
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            # Process Conv2d and Linear layers (excluding fc head)
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                # Get progenitor weight
                w_init = progenitor_state[f"{name}.weight"].to(device)
                
                # Weight has shape (C_out, C_in, ...)
                C_out = module.weight.shape[0]
                
                # Construct new weight channel-by-channel
                new_weight = torch.zeros_like(module.weight)
                scale_factors = []
                avg_sims = []
                
                for c in range(C_out):
                    w_init_c = w_init[c]
                    t_experts_c = [exp_mod.weight[c] - w_init_c for exp_mod in expert_modules]
                    t_merged_c = module.weight[c] - w_init_c
                    
                    # Pairwise cosine similarities of filter updates for this channel
                    pairwise_sims_c = []
                    for i in range(K):
                        for j in range(i + 1, K):
                            dot_prod = torch.sum(t_experts_c[i] * t_experts_c[j])
                            norm_i = torch.norm(t_experts_c[i])
                            norm_j = torch.norm(t_experts_c[j])
                            sim = dot_prod / (norm_i * norm_j + 1e-8)
                            pairwise_sims_c.append(sim)
                    
                    avg_sim_c = torch.mean(torch.stack(pairwise_sims_c)) if len(pairwise_sims_c) > 0 else torch.tensor(0.0).to(device)
                    avg_sims.append(avg_sim_c)
                    
                    # Compute channel-level lambda
                    sum_t_experts_c = torch.stack(t_experts_c).sum(dim=0)
                    norm_sum_t_c = torch.norm(sum_t_experts_c, p="fro")
                    norm_t_merged_c = torch.norm(t_merged_c, p="fro")
                    lam_c = norm_t_merged_c / (norm_sum_t_c + 1e-8)
                    
                    # Theoretical scale: S_theo = 1 / (lam_c * sqrt(K) * sqrt(1 + (2/K) * sum(rho_ij_c)))
                    sum_rho_c = torch.sum(torch.stack(pairwise_sims_c)) if len(pairwise_sims_c) > 0 else torch.tensor(0.0).to(device)
                    denom_c = lam_c * torch.sqrt(torch.tensor(float(K)).to(device)) * torch.sqrt(1.0 + (2.0 / K) * sum_rho_c)
                    
                    S_c = 1.0 / (denom_c + 1e-8)
                    S_c = torch.clamp(S_c, min=0.1, max=10.0)
                    scale_factors.append(S_c)
                    
                    new_weight[c] = w_init_c + S_c * t_merged_c
                    
                module.weight.copy_(new_weight)
                
                # Process bias if it exists
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    
                    new_bias = torch.zeros_like(module.bias)
                    for c in range(C_out):
                        S_c = scale_factors[c]
                        new_bias[c] = b_init[c] + S_c * b_merged[c]
                    module.bias.copy_(new_bias)
                    
                avg_S = torch.mean(torch.stack(scale_factors))
                avg_sim = torch.mean(torch.stack(avg_sims))
                print(f"CO-IPR Layer: {name:30s} | C_out: {C_out:4d} | Avg Cos Sim: {avg_sim.item():8.5f} | Avg S_theo_c: {avg_S.item():.4f}")
                
            # Process BatchNorm layers (scale BN weights/biases channel-wise)
            elif isinstance(module, nn.BatchNorm2d) and scale_bn:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                num_features = module.weight.shape[0]
                
                # Gamma weight (CU-IPR fallback)
                w_init = progenitor_state[f"{name}.weight"].to(device)
                t_experts = [exp_mod.weight - w_init for exp_mod in expert_modules]
                t_merged = module.weight - w_init
                
                new_weight = torch.zeros_like(module.weight)
                for c in range(num_features):
                    t_experts_c = [t_exp[c] for t_exp in t_experts]
                    t_merged_c = t_merged[c]
                    
                    norm_t_merged_c = torch.abs(t_merged_c)
                    norms_t_experts_c = torch.tensor([torch.abs(t_exp) for t_exp in t_experts_c])
                    avg_norm_t_experts_c = torch.mean(norms_t_experts_c)
                    
                    S_c = avg_norm_t_experts_c / (norm_t_merged_c + 1e-8)
                    S_c = torch.clamp(S_c, min=0.1, max=10.0)
                    new_weight[c] = w_init[c] + S_c * t_merged_c
                module.weight.copy_(new_weight)
                
                # Beta bias (CU-IPR fallback)
                if module.bias is not None:
                    b_init = progenitor_state[f"{name}.bias"].to(device)
                    b_experts = [exp_mod.bias - b_init for exp_mod in expert_modules]
                    b_merged = module.bias - b_init
                    
                    new_bias = torch.zeros_like(module.bias)
                    for c in range(num_features):
                        b_experts_c = [b_exp[c] for b_exp in b_experts]
                        b_merged_c = b_merged[c]
                        
                        norm_b_merged_c = torch.abs(b_merged_c)
                        norms_b_experts_c = torch.tensor([torch.abs(b_exp) for b_exp in b_experts_c])
                        avg_norm_b_experts_c = torch.mean(norms_b_experts_c)
                        
                        S_c = avg_norm_b_experts_c / (norm_b_merged_c + 1e-8)
                        S_c = torch.clamp(S_c, min=0.1, max=10.0)
                        new_bias[c] = b_init[c] + S_c * b_merged_c
                    module.bias.copy_(new_bias)
                    
    print("CO-IPR completed.")
    return merged_model


# -------------------------------------------------------------
# 6. Running Merging and Calibration Sweeps
# -------------------------------------------------------------
results_table = []

# --- 6.1 Weight Averaging (WA) Experiments ---
print("\n=======================================================")
print("RUNNING WEIGHT AVERAGING (WA) EXPERIMENTS")
print("=======================================================")

# Get standard WA model
wa_base = get_standard_merge(experts, progenitor_state, merge_type="WA")

# Evaluation: Uncalibrated
print("\n--- Evaluating WA (Uncalibrated) ---")
res_wa_uncal = evaluate_model(wa_base, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "None (Uncalibrated)",
    "mnist": res_wa_uncal["mnist"],
    "fmnist": res_wa_uncal["fmnist"],
    "cifar10": res_wa_uncal["cifar10"],
    "average": res_wa_uncal["average"]
})

# Evaluation: Corrected SP-TAAC
wa_sptaac = apply_sp_taac(wa_base, experts, train_loaders, num_samples=128)
print("\n--- Evaluating WA + SP-TAAC (Real-data Calibration) ---")
res_wa_sptaac = evaluate_model(wa_sptaac, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "SP-TAAC (Real N=128)",
    "mnist": res_wa_sptaac["mnist"],
    "fmnist": res_wa_sptaac["fmnist"],
    "cifar10": res_wa_sptaac["cifar10"],
    "average": res_wa_sptaac["average"]
})

# Evaluation: Weight-level IPR (Ours, Data-Free)
wa_w_ipr = apply_weight_level_ipr(wa_base, experts)
print("\n--- Evaluating WA + W-IPR (Ours, Data-Free) ---")
res_wa_w_ipr = evaluate_model(wa_w_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "W-IPR (Ours, Data-Free)",
    "mnist": res_wa_w_ipr["mnist"],
    "fmnist": res_wa_w_ipr["fmnist"],
    "cifar10": res_wa_w_ipr["cifar10"],
    "average": res_wa_w_ipr["average"]
})

# Evaluation: BN-level IPR (Ours, Data-Free)
wa_bn_ipr = apply_bn_level_ipr(wa_base, experts)
print("\n--- Evaluating WA + BN-IPR (Ours, Data-Free) ---")
res_wa_bn_ipr = evaluate_model(wa_bn_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "BN-IPR (Ours, Data-Free)",
    "mnist": res_wa_bn_ipr["mnist"],
    "fmnist": res_wa_bn_ipr["fmnist"],
    "cifar10": res_wa_bn_ipr["cifar10"],
    "average": res_wa_bn_ipr["average"]
})

# Evaluation: Update-level IPR (Ours, Data-Free)
wa_u_ipr = apply_update_level_ipr(wa_base, experts, progenitor_state)
print("\n--- Evaluating WA + U-IPR (Ours, Data-Free) ---")
res_wa_u_ipr = evaluate_model(wa_u_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "U-IPR (Ours, Data-Free)",
    "mnist": res_wa_u_ipr["mnist"],
    "fmnist": res_wa_u_ipr["fmnist"],
    "cifar10": res_wa_u_ipr["cifar10"],
    "average": res_wa_u_ipr["average"]
})

# Evaluation: Spectral-level IPR (Ours, Data-Free)
wa_s_ipr = apply_spectral_parameter_resonance(wa_base, experts, progenitor_state)
print("\n--- Evaluating WA + S-IPR (Ours, Data-Free) ---")
res_wa_s_ipr = evaluate_model(wa_s_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "S-IPR (Ours, Data-Free)",
    "mnist": res_wa_s_ipr["mnist"],
    "fmnist": res_wa_s_ipr["fmnist"],
    "cifar10": res_wa_s_ipr["cifar10"],
    "average": res_wa_s_ipr["average"]
})

# Evaluation: Subspace-Aligned IPR (Ours, Data-Free)
for alpha in [0.3, 0.5, 0.7]:
    wa_sa_ipr = apply_subspace_aligned_ipr(wa_base, experts, progenitor_state, alpha=alpha)
    print(f"\n--- Evaluating WA + SA-IPR (Ours, Data-Free, alpha={alpha}) ---")
    res_wa_sa_ipr = evaluate_model(wa_sa_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": "WA",
        "calibration": f"SA-IPR (Ours, alpha={alpha})",
        "mnist": res_wa_sa_ipr["mnist"],
        "fmnist": res_wa_sa_ipr["fmnist"],
        "cifar10": res_wa_sa_ipr["cifar10"],
        "average": res_wa_sa_ipr["average"]
    })

# Evaluation: Saliency-Weighted IPR (Ours, Data-Free)
wa_i_ipr = apply_saliency_weighted_ipr(wa_base, experts, progenitor_state)
print("\n--- Evaluating WA + I-IPR (Ours, Data-Free) ---")
res_wa_i_ipr = evaluate_model(wa_i_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "I-IPR (Ours, Data-Free)",
    "mnist": res_wa_i_ipr["mnist"],
    "fmnist": res_wa_i_ipr["fmnist"],
    "cifar10": res_wa_i_ipr["cifar10"],
    "average": res_wa_i_ipr["average"]
})

# Evaluation: Orthogonality-Aware IPR (Ours, Data-Free)
wa_oa_ipr = apply_orthogonality_aware_ipr(wa_base, experts, progenitor_state)
print("\n--- Evaluating WA + OA-IPR (Ours, Data-Free) ---")
res_wa_oa_ipr = evaluate_model(wa_oa_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "OA-IPR (Ours, Data-Free)",
    "mnist": res_wa_oa_ipr["mnist"],
    "fmnist": res_wa_oa_ipr["fmnist"],
    "cifar10": res_wa_oa_ipr["cifar10"],
    "average": res_wa_oa_ipr["average"]
})

# Evaluation: Channel-level Update IPR (Ours, Data-Free)
wa_cu_ipr = apply_channel_update_level_ipr(wa_base, experts, progenitor_state)
print("\n--- Evaluating WA + CU-IPR (Ours, Data-Free) ---")
res_wa_cu_ipr = evaluate_model(wa_cu_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "CU-IPR (Ours, Data-Free)",
    "mnist": res_wa_cu_ipr["mnist"],
    "fmnist": res_wa_cu_ipr["fmnist"],
    "cifar10": res_wa_cu_ipr["cifar10"],
    "average": res_wa_cu_ipr["average"]
})

# Evaluation: Channel-level Orthogonality-Aware IPR (Ours, Data-Free)
wa_co_ipr = apply_channel_orthogonality_aware_ipr(wa_base, experts, progenitor_state)
print("\n--- Evaluating WA + CO-IPR (Ours, Data-Free) ---")
res_wa_co_ipr = evaluate_model(wa_co_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "CO-IPR (Ours, Data-Free)",
    "mnist": res_wa_co_ipr["mnist"],
    "fmnist": res_wa_co_ipr["fmnist"],
    "cifar10": res_wa_co_ipr["cifar10"],
    "average": res_wa_co_ipr["average"]
})


# --- 6.2 Task Arithmetic (TA) Experiments ---
print("\n=======================================================")
print("RUNNING TASK ARITHMETIC (TA) EXPERIMENTS")
print("=======================================================")

for lam in [0.2, 0.5]:
    print(f"\n--- TA with lambda = {lam} ---")
    ta_base = get_standard_merge(experts, progenitor_state, merge_type="TA", lam=lam)
    
    # Evaluation: Uncalibrated
    print(f"\n--- Evaluating TA (lambda={lam}, Uncalibrated) ---")
    res_ta_uncal = evaluate_model(ta_base, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "None (Uncalibrated)",
        "mnist": res_ta_uncal["mnist"],
        "fmnist": res_ta_uncal["fmnist"],
        "cifar10": res_ta_uncal["cifar10"],
        "average": res_ta_uncal["average"]
    })
    
    # Evaluation: Corrected SP-TAAC
    ta_sptaac = apply_sp_taac(ta_base, experts, train_loaders, num_samples=128)
    print(f"\n--- Evaluating TA (lambda={lam}) + SP-TAAC ---")
    res_ta_sptaac = evaluate_model(ta_sptaac, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "SP-TAAC (Real N=128)",
        "mnist": res_ta_sptaac["mnist"],
        "fmnist": res_ta_sptaac["fmnist"],
        "cifar10": res_ta_sptaac["cifar10"],
        "average": res_ta_sptaac["average"]
    })
    
    # Evaluation: Weight-level IPR (Ours, Data-Free)
    ta_w_ipr = apply_weight_level_ipr(ta_base, experts)
    print(f"\n--- Evaluating TA (lambda={lam}) + W-IPR (Ours, Data-Free) ---")
    res_ta_w_ipr = evaluate_model(ta_w_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "W-IPR (Ours, Data-Free)",
        "mnist": res_ta_w_ipr["mnist"],
        "fmnist": res_ta_w_ipr["fmnist"],
        "cifar10": res_ta_w_ipr["cifar10"],
        "average": res_ta_w_ipr["average"]
    })
    
    # Evaluation: BN-level IPR (Ours, Data-Free)
    ta_bn_ipr = apply_bn_level_ipr(ta_base, experts)
    print(f"\n--- Evaluating TA (lambda={lam}) + BN-IPR (Ours, Data-Free) ---")
    res_ta_bn_ipr = evaluate_model(ta_bn_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "BN-IPR (Ours, Data-Free)",
        "mnist": res_ta_bn_ipr["mnist"],
        "fmnist": res_ta_bn_ipr["fmnist"],
        "cifar10": res_ta_bn_ipr["cifar10"],
        "average": res_ta_bn_ipr["average"]
    })

    # Evaluation: Update-level IPR (Ours, Data-Free)
    ta_u_ipr = apply_update_level_ipr(ta_base, experts, progenitor_state)
    print(f"\n--- Evaluating TA (lambda={lam}) + U-IPR (Ours, Data-Free) ---")
    res_ta_u_ipr = evaluate_model(ta_u_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "U-IPR (Ours, Data-Free)",
        "mnist": res_ta_u_ipr["mnist"],
        "fmnist": res_ta_u_ipr["fmnist"],
        "cifar10": res_ta_u_ipr["cifar10"],
        "average": res_ta_u_ipr["average"]
    })

    # Evaluation: Spectral-level IPR (Ours, Data-Free)
    ta_s_ipr = apply_spectral_parameter_resonance(ta_base, experts, progenitor_state)
    print(f"\n--- Evaluating TA (lambda={lam}) + S-IPR (Ours, Data-Free) ---")
    res_ta_s_ipr = evaluate_model(ta_s_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "S-IPR (Ours, Data-Free)",
        "mnist": res_ta_s_ipr["mnist"],
        "fmnist": res_ta_s_ipr["fmnist"],
        "cifar10": res_ta_s_ipr["cifar10"],
        "average": res_ta_s_ipr["average"]
    })

    # Evaluation: Subspace-Aligned IPR (Ours, Data-Free)
    for alpha in [0.3, 0.5, 0.7]:
        ta_sa_ipr = apply_subspace_aligned_ipr(ta_base, experts, progenitor_state, alpha=alpha)
        print(f"\n--- Evaluating TA (lambda={lam}) + SA-IPR (Ours, Data-Free, alpha={alpha}) ---")
        res_ta_sa_ipr = evaluate_model(ta_sa_ipr, experts, test_loaders)
        results_table.append({
            "merge_type": f"TA (lam={lam})",
            "calibration": f"SA-IPR (Ours, alpha={alpha})",
            "mnist": res_ta_sa_ipr["mnist"],
            "fmnist": res_ta_sa_ipr["fmnist"],
            "cifar10": res_ta_sa_ipr["cifar10"],
            "average": res_ta_sa_ipr["average"]
        })

    # Evaluation: Saliency-Weighted IPR (Ours, Data-Free)
    ta_i_ipr = apply_saliency_weighted_ipr(ta_base, experts, progenitor_state)
    print(f"\n--- Evaluating TA (lambda={lam}) + I-IPR (Ours, Data-Free) ---")
    res_ta_i_ipr = evaluate_model(ta_i_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "I-IPR (Ours, Data-Free)",
        "mnist": res_ta_i_ipr["mnist"],
        "fmnist": res_ta_i_ipr["fmnist"],
        "cifar10": res_ta_i_ipr["cifar10"],
        "average": res_ta_i_ipr["average"]
    })

    # Evaluation: Orthogonality-Aware IPR (Ours, Data-Free)
    ta_oa_ipr = apply_orthogonality_aware_ipr(ta_base, experts, progenitor_state)
    print(f"\n--- Evaluating TA (lambda={lam}) + OA-IPR (Ours, Data-Free) ---")
    res_ta_oa_ipr = evaluate_model(ta_oa_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "OA-IPR (Ours, Data-Free)",
        "mnist": res_ta_oa_ipr["mnist"],
        "fmnist": res_ta_oa_ipr["fmnist"],
        "cifar10": res_ta_oa_ipr["cifar10"],
        "average": res_ta_oa_ipr["average"]
    })

    # Evaluation: Channel-level Update IPR (Ours, Data-Free)
    ta_cu_ipr = apply_channel_update_level_ipr(ta_base, experts, progenitor_state)
    print(f"\n--- Evaluating TA (lambda={lam}) + CU-IPR (Ours, Data-Free) ---")
    res_ta_cu_ipr = evaluate_model(ta_cu_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "CU-IPR (Ours, Data-Free)",
        "mnist": res_ta_cu_ipr["mnist"],
        "fmnist": res_ta_cu_ipr["fmnist"],
        "cifar10": res_ta_cu_ipr["cifar10"],
        "average": res_ta_cu_ipr["average"]
    })

    # Evaluation: Channel-level Orthogonality-Aware IPR (Ours, Data-Free)
    ta_co_ipr = apply_channel_orthogonality_aware_ipr(ta_base, experts, progenitor_state)
    print(f"\n--- Evaluating TA (lambda={lam}) + CO-IPR (Ours, Data-Free) ---")
    res_ta_co_ipr = evaluate_model(ta_co_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "CO-IPR (Ours, Data-Free)",
        "mnist": res_ta_co_ipr["mnist"],
        "fmnist": res_ta_co_ipr["fmnist"],
        "cifar10": res_ta_co_ipr["cifar10"],
        "average": res_ta_co_ipr["average"]
    })

# Add Oracle results for context
results_table.append({
    "merge_type": "Oracle Experts (Individual)",
    "calibration": "None",
    "mnist": oracle_results["mnist"],
    "fmnist": oracle_results["fmnist"],
    "cifar10": oracle_results["cifar10"],
    "average": oracle_results["average"]
})

# -------------------------------------------------------------
# 7. Print and Save Results
# -------------------------------------------------------------
print("\n=======================================================")
print("FINAL ACCURACY RESULTS SUMMARY TABLE")
print("=======================================================")
print(f"{'Merge Method':25s} | {'Calibration':25s} | {'MNIST':6s} | {'F-MNIST':6s} | {'CIFAR10':6s} | {'Average':6s}")
print("-" * 88)
for row in results_table:
    print(f"{row['merge_type']:25s} | {row['calibration']:25s} | {row['mnist']:5.2f}% | {row['fmnist']:5.2f}% | {row['cifar10']:5.2f}% | {row['average']:5.2f}%")
print("=======================================================")

# Save to JSON
with open("results.json", "w") as f:
    json.dump(results_table, f, indent=4)
print("Saved final results to results.json.")
