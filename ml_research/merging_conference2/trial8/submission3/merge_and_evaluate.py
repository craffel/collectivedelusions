import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from train import get_dataset, evaluate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

def load_model_from_state_dict(state_dict_path, device):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model = model.to(device)
    return model

def get_backbone_and_head(state_dict):
    """Splits a state_dict into backbone and classification head (fc)."""
    backbone = {}
    head = {}
    for k, v in state_dict.items():
        if k.startswith("fc."):
            head[k] = v.clone()
        else:
            backbone[k] = v.clone()
    return backbone, head

def compute_l2_norm_dim0(tensor):
    """Computes the L2 norm along the 0th dimension (channel-wise)."""
    if tensor.dim() == 1:
        return torch.abs(tensor)
    else:
        # Flatten all dimensions except the first one and compute norm
        flat_tensor = tensor.flatten(start_dim=1)
        return torch.norm(flat_tensor, p=2, dim=1)

def merge_weight_averaging(expert_backbones):
    """Standard Weight Averaging (WA) of backbones."""
    merged_backbone = {}
    keys = expert_backbones[0].keys()
    K = len(expert_backbones)
    for k in keys:
        tensor_list = [expert[k] for expert in expert_backbones]
        # For integer/boolean buffers, use the first expert's buffer
        if not tensor_list[0].is_floating_point():
            merged_backbone[k] = tensor_list[0].clone()
        else:
            merged_backbone[k] = torch.stack(tensor_list, dim=0).mean(dim=0)
    return merged_backbone

def merge_task_arithmetic(progenitor_backbone, expert_backbones, lmbda):
    """Task Arithmetic (TA) merging: W_init + lambda * sum(T_k)."""
    merged_backbone = {}
    keys = progenitor_backbone.keys()
    for k in keys:
        prog_tensor = progenitor_backbone[k]
        if not prog_tensor.is_floating_point():
            merged_backbone[k] = prog_tensor.clone()
            continue
            
        task_vectors = [expert[k] - prog_tensor for expert in expert_backbones]
        merged_vector = torch.stack(task_vectors, dim=0).sum(dim=0)
        merged_backbone[k] = prog_tensor + lmbda * merged_vector
    return merged_backbone

def merge_u_ipr(progenitor_backbone, expert_backbones, clamp_min=0.1, clamp_max=10.0):
    """Update-level Isotropic Parameter Resonance (U-IPR) layer-wise scaling."""
    merged_backbone = {}
    keys = progenitor_backbone.keys()
    K = len(expert_backbones)
    
    # First compute standard WA backbone
    wa_backbone = merge_weight_averaging(expert_backbones)
    
    for k in keys:
        prog_tensor = progenitor_backbone[k]
        wa_tensor = wa_backbone[k]
        
        if not prog_tensor.is_floating_point() or "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            merged_backbone[k] = wa_tensor.clone()
            continue
            
        # Task vectors
        task_vectors = [expert[k] - prog_tensor for expert in expert_backbones]
        merged_vector = wa_tensor - prog_tensor
        
        # Compute layer-wise Frobenius norms
        expert_norms = [torch.norm(tv, p="fro") for tv in task_vectors]
        avg_expert_norm = torch.stack(expert_norms).mean()
        merged_norm = torch.norm(merged_vector, p="fro")
        
        # Scaling factor
        S_l = avg_expert_norm / (merged_norm + 1e-8)
        S_l = torch.clamp(S_l, clamp_min, clamp_max)
        
        # Rescale merged task vector and reconstruct weights
        merged_backbone[k] = prog_tensor + S_l * merged_vector
        
    return merged_backbone

def merge_hns(progenitor_backbone, expert_backbones, task_idx, clamp_min=0.1, clamp_max=10.0):
    """Holographic Norm Scaling (HNS) - reconstructs weights for specific task_idx."""
    merged_backbone = {}
    keys = progenitor_backbone.keys()
    K = len(expert_backbones)
    
    # Target expert and its task vector
    target_expert = expert_backbones[task_idx]
    
    # Standard Weight Averaging backbone
    wa_backbone = merge_weight_averaging(expert_backbones)
    
    for k in keys:
        prog_tensor = progenitor_backbone[k]
        wa_tensor = wa_backbone[k]
        
        # If it's not a parameter (e.g., BatchNorm running statistics), HNS simply swaps in
        # the original target expert's buffers (as described in submission7 Section 3.3).
        if not prog_tensor.is_floating_point() or "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            merged_backbone[k] = target_expert[k].clone()
            continue
            
        # If it is a scale/bias of BatchNorm, HNS also swaps in target expert's parameters
        if "weight" in k and ("bn" in k or "downsample.1" in k):
            merged_backbone[k] = target_expert[k].clone()
            continue
        if "bias" in k and ("bn" in k or "downsample.1" in k):
            merged_backbone[k] = target_expert[k].clone()
            continue
            
        # For Conv2d and Linear layers, apply HNS channel-wise scaling
        expert_vector = target_expert[k] - prog_tensor
        merged_vector = wa_tensor - prog_tensor
        
        # Compute channel-wise (dim 0) L2 norms
        expert_norms = compute_l2_norm_dim0(expert_vector)
        merged_norms = compute_l2_norm_dim0(merged_vector)
        
        # Compute channel-wise scale factors
        gamma = expert_norms / (merged_norms + 1e-8)
        gamma = torch.clamp(gamma, clamp_min, clamp_max)
        
        # Apply scaling to the merged vector along dimension 0
        scaled_merged_vector = merged_vector.clone()
        for c in range(scaled_merged_vector.size(0)):
            scaled_merged_vector[c] = scaled_merged_vector[c] * gamma[c]
            
        # Reconstruct weights
        merged_backbone[k] = prog_tensor + scaled_merged_vector
        
    return merged_backbone

def merge_ucpr(progenitor_backbone, expert_backbones, clamp_min=0.1, clamp_max=10.0):
    """Unified Channel-wise Parameter Resonance (UCPR) - Ours!"""
    merged_backbone = {}
    keys = progenitor_backbone.keys()
    K = len(expert_backbones)
    
    # Standard Weight Averaging backbone
    wa_backbone = merge_weight_averaging(expert_backbones)
    
    for k in keys:
        prog_tensor = progenitor_backbone[k]
        wa_tensor = wa_backbone[k]
        
        # If it's not a floating point or is a running buffer, use averaged WA
        if not prog_tensor.is_floating_point() or "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            merged_backbone[k] = wa_tensor.clone()
            continue
            
        # Task vectors
        task_vectors = [expert[k] - prog_tensor for expert in expert_backbones]
        merged_vector = wa_tensor - prog_tensor
        
        # Compute channel-wise (dim 0) L2 norms for all experts
        expert_norms_list = [compute_l2_norm_dim0(tv) for tv in task_vectors]
        # Average of expert norms channel-wise
        avg_expert_norm = torch.stack(expert_norms_list, dim=0).mean(dim=0)
        
        # Merged norm channel-wise
        merged_norm = compute_l2_norm_dim0(merged_vector)
        
        # Channel-wise scale factor
        S_lc = avg_expert_norm / (merged_norm + 1e-8)
        S_lc = torch.clamp(S_lc, clamp_min, clamp_max)
        
        # Apply scaling channel-by-channel along dimension 0
        scaled_merged_vector = merged_vector.clone()
        for c in range(scaled_merged_vector.size(0)):
            scaled_merged_vector[c] = scaled_merged_vector[c] * S_lc[c]
            
        # Reconstruct weights
        merged_backbone[k] = prog_tensor + scaled_merged_vector
        
    return merged_backbone

def merge_depth_adaptive_hpr(progenitor_backbone, expert_backbones, alpha_shallow, alpha_deep, clamp_min=0.1, clamp_max=10.0):
    """Depth-Adaptive Hybrid Parameter Resonance (DA-HPR)"""
    merged_backbone = {}
    keys = list(progenitor_backbone.keys())
    
    # Filter keys that are float parameters needing scaling
    valid_keys = []
    for k in keys:
        tensor = progenitor_backbone[k]
        if tensor.is_floating_point() and "running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k:
            valid_keys.append(k)
            
    N = len(valid_keys)
    
    # Standard Weight Averaging backbone
    wa_backbone = merge_weight_averaging(expert_backbones)
    
    for k in keys:
        prog_tensor = progenitor_backbone[k]
        wa_tensor = wa_backbone[k]
        
        # If it's not in valid_keys, use averaged WA
        if k not in valid_keys:
            merged_backbone[k] = wa_tensor.clone()
            continue
            
        # Compute layer depth d_l in [0, 1]
        idx = valid_keys.index(k)
        d_l = idx / max(1, N - 1)
        
        # Linear interpolation of alpha based on depth
        alpha_l = (1.0 - d_l) * alpha_shallow + d_l * alpha_deep
        
        # Task vectors
        task_vectors = [expert[k] - prog_tensor for expert in expert_backbones]
        merged_vector = wa_tensor - prog_tensor
        
        # 1. Compute layer-wise (Frobenius) scaling factor S_l
        expert_norms_fro = [torch.norm(tv, p="fro") for tv in task_vectors]
        avg_expert_norm_fro = torch.stack(expert_norms_fro).mean()
        merged_norm_fro = torch.norm(merged_vector, p="fro")
        S_l = avg_expert_norm_fro / (merged_norm_fro + 1e-8)
        S_l = torch.clamp(S_l, clamp_min, clamp_max)
        
        # 2. Compute channel-wise (dim 0) scaling factor S_lc
        expert_norms_list = [compute_l2_norm_dim0(tv) for tv in task_vectors]
        avg_expert_norm_c = torch.stack(expert_norms_list, dim=0).mean(dim=0)
        merged_norm_c = compute_l2_norm_dim0(merged_vector)
        S_lc = avg_expert_norm_c / (merged_norm_c + 1e-8)
        S_lc = torch.clamp(S_lc, clamp_min, clamp_max)
        
        # 3. Dynamic hybrid scaling factor
        S_hybrid = alpha_l * S_lc + (1.0 - alpha_l) * S_l
        S_hybrid = torch.clamp(S_hybrid, clamp_min, clamp_max)
        
        # Apply scaling channel-by-channel along dimension 0
        scaled_merged_vector = merged_vector.clone()
        for c in range(scaled_merged_vector.size(0)):
            scaled_merged_vector[c] = scaled_merged_vector[c] * S_hybrid[c]
            
        # Reconstruct weights
        merged_backbone[k] = prog_tensor + scaled_merged_vector
        
    return merged_backbone

def create_full_model(backbone_dict, head_dict, device):
    """Combines backbone and head state dicts into a full model."""
    full_state_dict = {**backbone_dict, **head_dict}
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(full_state_dict)
    model = model.to(device)
    model.eval()
    return model

def calibrate_batchnorm(backbone, heads, train_loaders, device, num_samples_per_task=512, batch_size=64):
    """Calibrates the BatchNorm statistics of the merged backbone on a mixed dataset."""
    first_task = list(heads.keys())[0]
    cal_model = create_full_model(backbone, heads[first_task], device)
    
    cal_model.train()
    
    for m in cal_model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = 0.1
            
    mixed_inputs = []
    for task_name, loader in train_loaders.items():
        dataset = loader.dataset
        num_samples = min(num_samples_per_task, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)
        for idx in indices:
            img, _ = dataset[idx]
            mixed_inputs.append(img)
            
    random.shuffle(mixed_inputs)
    cal_loader = DataLoader(mixed_inputs, batch_size=batch_size, shuffle=True)
    
    with torch.no_grad():
        for epoch in range(5):
            for x in cal_loader:
                x = x.to(device)
                _ = cal_model(x)
                
    cal_backbone = {}
    for k, v in cal_model.state_dict().items():
        if not k.startswith("fc."):
            cal_backbone[k] = v.clone()
            
    return cal_backbone

def main():
    parser = argparse.ArgumentParser(description="Merge and Evaluate Experts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths to checkpoints
    prog_path = "progenitor.pt"
    mnist_path = "mnist_expert.pt"
    fmnist_path = "fmnist_expert.pt"
    cifar10_path = "cifar10_expert.pt"
    
    # Verify checkponts exist
    for p in [prog_path, mnist_path, fmnist_path, cifar10_path]:
        if not os.path.exists(p):
            print(f"Error: checkpoint {p} not found. Make sure train_experts job is finished.")
            return
            
    # Load state dicts
    prog_sd = torch.load(prog_path, map_location=device)
    mnist_sd = torch.load(mnist_path, map_location=device)
    fmnist_sd = torch.load(fmnist_path, map_location=device)
    cifar10_sd = torch.load(cifar10_path, map_location=device)
    
    # Separate backbones and heads
    prog_backbone, _ = get_backbone_and_head(prog_sd)
    mnist_backbone, mnist_head = get_backbone_and_head(mnist_sd)
    fmnist_backbone, fmnist_head = get_backbone_and_head(fmnist_sd)
    cifar10_backbone, cifar10_head = get_backbone_and_head(cifar10_sd)
    
    expert_backbones = [mnist_backbone, fmnist_backbone, cifar10_backbone]
    heads = {
        "mnist": mnist_head,
        "fmnist": fmnist_head,
        "cifar10": cifar10_head
    }
    
    # Load datasets
    print("Loading datasets...")
    mnist_train, mnist_test = get_dataset("mnist", download=False)
    fmnist_train, fmnist_test = get_dataset("fmnist", download=False)
    cifar10_train, cifar10_test = get_dataset("cifar10", download=False)
    
    test_loaders = {
        "mnist": DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False, num_workers=4),
        "fmnist": DataLoader(fmnist_test, batch_size=args.batch_size, shuffle=False, num_workers=4),
        "cifar10": DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    }
    
    train_loaders = {
        "mnist": DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, num_workers=4),
        "fmnist": DataLoader(fmnist_train, batch_size=args.batch_size, shuffle=True, num_workers=4),
        "cifar10": DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    }
    
    results_no_cal = {}
    results_cal = {}
    
    # 1. Evaluate Oracles (Upper Bounds)
    print("\n--- Evaluating Expert Oracles ---")
    mnist_oracle = create_full_model(mnist_backbone, mnist_head, device)
    fmnist_oracle = create_full_model(fmnist_backbone, fmnist_head, device)
    cifar10_oracle = create_full_model(cifar10_backbone, cifar10_head, device)
    
    mnist_oracle_acc = evaluate(mnist_oracle, test_loaders["mnist"], device)
    fmnist_oracle_acc = evaluate(fmnist_oracle, test_loaders["fmnist"], device)
    cifar10_oracle_acc = evaluate(cifar10_oracle, test_loaders["cifar10"], device)
    
    print(f"MNIST Oracle: {mnist_oracle_acc:.2f}%")
    print(f"Fashion-MNIST Oracle: {fmnist_oracle_acc:.2f}%")
    print(f"CIFAR-10 Oracle: {cifar10_oracle_acc:.2f}%")
    oracle_avg = np.mean([mnist_oracle_acc, fmnist_oracle_acc, cifar10_oracle_acc])
    print(f"Oracle Average: {oracle_avg:.2f}%")
    
    # Define merged backbones to evaluate
    backbones = {
        "WA": merge_weight_averaging(expert_backbones),
        "TA (lambda=0.3)": merge_task_arithmetic(prog_backbone, expert_backbones, 0.3),
        "U-IPR": merge_u_ipr(prog_backbone, expert_backbones),
        "HNS (task-agnostic)": merge_hns(prog_backbone, expert_backbones, 0), # evaluate task-agnostically with task 0
        "UCPR": merge_ucpr(prog_backbone, expert_backbones),
        "DA-HPR (S=1.0, D=0.1)": merge_depth_adaptive_hpr(prog_backbone, expert_backbones, 1.0, 0.1),
        "DA-HPR (S=0.6, D=0.4)": merge_depth_adaptive_hpr(prog_backbone, expert_backbones, 0.6, 0.4)
    }
    
    print("\n--- Evaluating Merging Methods WITHOUT BatchNorm Calibration ---")
    for name, backbone in backbones.items():
        accs = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            model = create_full_model(backbone, heads[task], device)
            accs[task] = evaluate(model, test_loaders[task], device)
        results_no_cal[name] = accs
        avg = np.mean(list(accs.values()))
        print(f"{name:<22} MNIST: {accs['mnist']:7.2f}% | F-MNIST: {accs['fmnist']:7.2f}% | CIFAR-10: {accs['cifar10']:7.2f}% | Average: {avg:7.2f}%")
        
    print("\n--- Evaluating Merging Methods WITH BatchNorm Calibration ---")
    for name, backbone in backbones.items():
        # Perform BNC calibration
        cal_backbone = calibrate_batchnorm(backbone, heads, train_loaders, device, num_samples_per_task=512, batch_size=64)
        accs = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            model = create_full_model(cal_backbone, heads[task], device)
            accs[task] = evaluate(model, test_loaders[task], device)
        results_cal[name] = accs
        avg = np.mean(list(accs.values()))
        print(f"{name:<22} MNIST: {accs['mnist']:7.2f}% | F-MNIST: {accs['fmnist']:7.2f}% | CIFAR-10: {accs['cifar10']:7.2f}% | Average: {avg:7.2f}%")
        
    # Summary Table
    print("\n" + "="*100)
    print(f"{'Method':<22} | {'Clean (No BNC)':<32} | {'Calibrated (With BNC)':<32}")
    print(f"{'':<22} | {'MNIST':<6} {'F-MN':<5} {'CIF10':<5} {'AVG':<6} | {'MNIST':<6} {'F-MN':<5} {'CIF10':<5} {'AVG':<6}")
    print("-"*100)
    
    # Print Oracle row
    print(f"{'Oracle':<22} | {mnist_oracle_acc:<6.2f} {fmnist_oracle_acc:<5.2f} {cifar10_oracle_acc:<5.2f} {oracle_avg:<6.2f} | {'N/A':<6} {'N/A':<5} {'N/A':<5} {'N/A':<6}")
    print("-"*100)
    
    for name in backbones.keys():
        nc = results_no_cal[name]
        c = results_cal[name]
        avg_nc = np.mean(list(nc.values()))
        avg_c = np.mean(list(c.values()))
        print(f"{name:<22} | {nc['mnist']:<6.2f} {nc['fmnist']:<5.2f} {nc['cifar10']:<5.2f} {avg_nc:<6.2f} | {c['mnist']:<6.2f} {c['fmnist']:<5.2f} {c['cifar10']:<5.2f} {avg_c:<6.2f}")
    print("="*100)

if __name__ == "__main__":
    main()
