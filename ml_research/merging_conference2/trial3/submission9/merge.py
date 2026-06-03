import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on cluster nodes
torch.backends.cudnn.enabled = False

def get_transforms():
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform_rgb, transform_gray

def get_dataset(task, is_train=False):
    transform_rgb, transform_gray = get_transforms()
    if task == "mnist":
        return datasets.MNIST(root="data", train=is_train, download=True, transform=transform_gray)
    elif task == "fashion":
        return datasets.FashionMNIST(root="data", train=is_train, download=True, transform=transform_gray)
    elif task == "cifar10":
        return datasets.CIFAR10(root="data", train=is_train, download=True, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown task: {task}")

def load_model(path=None):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

# 2D DCT Transform Helpers
def get_dct_matrix(size, device):
    i = torch.arange(size, device=device).unsqueeze(1)
    j = torch.arange(size, device=device).unsqueeze(0)
    angles = (np.pi * i * (2.0 * j + 1.0)) / (2.0 * size)
    C = torch.sqrt(torch.tensor(2.0 / size, device=device)) * torch.cos(angles)
    C[0, :] = 1.0 / np.sqrt(size)
    return C

def dct_2d(X):
    M, N = X.shape
    C_M = get_dct_matrix(M, X.device)
    C_N = get_dct_matrix(N, X.device)
    return C_M @ X @ C_N.t(), C_M, C_N

def idct_2d(D, C_M, C_N):
    return C_M.t() @ D @ C_N

def stdfs_merge_tensor(tensors, low_freq_ratio):
    # tensors: list of 3 tensors of the same shape
    # Returns the merged tensor
    device = tensors[0].device
    shape = tensors[0].shape
    
    # For 1D tensors, just do standard averaging
    if len(shape) < 2:
        return torch.stack(tensors, dim=0).mean(dim=0)
        
    # Reshape to 2D
    M = shape[0]
    N = 1
    for s in shape[1:]:
        N *= s
        
    reshaped_tensors = [t.view(M, N).float() for t in tensors]
    
    # Compute DCT for each task
    DCT_mats = []
    C_M, C_N = None, None
    for X in reshaped_tensors:
        D, C_M, C_N = dct_2d(X)
        DCT_mats.append(D)
        
    # Create frequency mask
    u_grid = torch.arange(M, device=device).unsqueeze(1).expand(M, N)
    v_grid = torch.arange(N, device=device).unsqueeze(0).expand(M, N)
    dist = u_grid.float() / M + v_grid.float() / N
    
    # Quantile threshold for low frequency split
    threshold = torch.quantile(dist, low_freq_ratio)
    low_mask = dist <= threshold
    
    # Low frequency merging (average)
    D_low = torch.stack(DCT_mats, dim=0).mean(dim=0)
    
    # High frequency merging (relative magnitude-based max pooling)
    mats_stack = torch.stack(DCT_mats, dim=0) # Shape: (3, M, N)
    
    # Normalize by Frobenius norm of each task vector to ensure balanced relative scaling
    norms = torch.norm(mats_stack, p='fro', dim=(1, 2), keepdim=True) + 1e-8
    normalized_mats = mats_stack / norms
    
    abs_stack = torch.abs(normalized_mats)
    max_idx = torch.argmax(abs_stack, dim=0) # Shape: (M, N)
    D_high = torch.gather(mats_stack, 0, max_idx.unsqueeze(0)).squeeze(0)
    
    # Combine using low_mask
    D_merged = torch.where(low_mask, D_low, D_high)
    
    # Inverse DCT
    X_merged = idct_2d(D_merged, C_M, C_N)
    return X_merged.view(shape).to(tensors[0].dtype)

def swsm_merge_tensor(tensors, low_freq_ratio, gamma=0.05):
    # Spectral Window-based Soft Merging (SWSM)
    # Smooths the boundary in the frequency domain using a sigmoid gating function,
    # preventing high-frequency Gibbs ringing and boundary reconstruction artifacts.
    device = tensors[0].device
    shape = tensors[0].shape
    
    # For 1D tensors, just do standard averaging
    if len(shape) < 2:
        return torch.stack(tensors, dim=0).mean(dim=0)
        
    # Reshape to 2D
    M = shape[0]
    N = 1
    for s in shape[1:]:
        N *= s
        
    reshaped_tensors = [t.view(M, N).float() for t in tensors]
    
    # Compute DCT for each task
    DCT_mats = []
    C_M, C_N = None, None
    for X in reshaped_tensors:
        D, C_M, C_N = dct_2d(X)
        DCT_mats.append(D)
        
    # Create frequency distance
    u_grid = torch.arange(M, device=device).unsqueeze(1).expand(M, N)
    v_grid = torch.arange(N, device=device).unsqueeze(0).expand(M, N)
    dist = u_grid.float() / M + v_grid.float() / N
    
    # Quantile threshold for low frequency split
    threshold = torch.quantile(dist, low_freq_ratio)
    
    # Sigmoid gating
    if gamma <= 0:
        gate = (dist <= threshold).float()
    else:
        gate = torch.sigmoid((threshold - dist) / gamma)
        
    # Low frequency merging (average)
    D_low = torch.stack(DCT_mats, dim=0).mean(dim=0)
    
    # High frequency merging (relative magnitude-based max pooling)
    mats_stack = torch.stack(DCT_mats, dim=0) # Shape: (3, M, N)
    norms = torch.norm(mats_stack, p='fro', dim=(1, 2), keepdim=True) + 1e-8
    normalized_mats = mats_stack / norms
    abs_stack = torch.abs(normalized_mats)
    max_idx = torch.argmax(abs_stack, dim=0)
    D_high = torch.gather(mats_stack, 0, max_idx.unsqueeze(0)).squeeze(0)
    
    # Soft blend
    D_merged = gate * D_low + (1.0 - gate) * D_high
    
    # Inverse DCT
    X_merged = idct_2d(D_merged, C_M, C_N)
    return X_merged.view(shape).to(tensors[0].dtype)

def dare_merge_tensor(tensors, drop_rate):
    device = tensors[0].device
    shape = tensors[0].shape
    
    if drop_rate <= 0:
        return torch.stack(tensors, dim=0).mean(dim=0)
        
    pruned_tensors = []
    for t in tensors:
        mask = (torch.rand_like(t.float()) >= drop_rate).to(device)
        pruned = (t.float() * mask) / (1.0 - drop_rate)
        pruned_tensors.append(pruned)
        
    return torch.stack(pruned_tensors, dim=0).mean(dim=0).to(tensors[0].dtype)

def ties_merge_tensor(tensors, keep_ratio):
    device = tensors[0].device
    shape = tensors[0].shape
    
    # Trim: for each tensor, keep only top keep_ratio by magnitude
    trimmed_tensors = []
    for t in tensors:
        t_flat = t.view(-1).float()
        num_keep = max(1, int(keep_ratio * t_flat.numel()))
        threshold = torch.topk(torch.abs(t_flat), num_keep).values[-1]
        mask = torch.abs(t) >= threshold
        trimmed_tensors.append(t.float() * mask)
        
    # Elect consensus sign
    stacked = torch.stack(trimmed_tensors, dim=0) # Shape: (K, *shape)
    pos_mask = stacked > 0
    neg_mask = stacked < 0
    
    pos_sum = (stacked * pos_mask).sum(dim=0)
    neg_sum = torch.abs((stacked * neg_mask).sum(dim=0))
    
    consensus_sign = torch.where(pos_sum >= neg_sum, torch.ones_like(pos_sum), -torch.ones_like(pos_sum))
    
    # Disagree check
    sign_stacked = torch.sign(stacked)
    agree_mask = (sign_stacked == consensus_sign.unsqueeze(0)) | (sign_stacked == 0)
    agreed_stacked = stacked * agree_mask
    
    # Average across non-zero elements
    non_zero_counts = (agreed_stacked != 0).sum(dim=0).float()
    sum_agreed = agreed_stacked.sum(dim=0)
    
    merged = torch.where(non_zero_counts > 0, sum_agreed / non_zero_counts, torch.zeros_like(sum_agreed))
    return merged.to(tensors[0].dtype)

def s_dare_merge_tensor(tensors, drop_rate):
    device = tensors[0].device
    shape = tensors[0].shape
    
    if len(shape) < 2:
        return dare_merge_tensor(tensors, drop_rate)
        
    M = shape[0]
    N = 1
    for s in shape[1:]:
        N *= s
        
    reshaped_tensors = [t.view(M, N).float() for t in tensors]
    
    # Transform to DCT
    DCT_mats = []
    C_M, C_N = None, None
    for X in reshaped_tensors:
        D, C_M, C_N = dct_2d(X)
        DCT_mats.append(D)
        
    # Apply DARE in DCT domain
    merged_dct = dare_merge_tensor(DCT_mats, drop_rate)
    
    # Inverse DCT
    X_merged = idct_2d(merged_dct, C_M, C_N)
    return X_merged.view(shape).to(tensors[0].dtype)

def s_ties_merge_tensor(tensors, keep_ratio):
    device = tensors[0].device
    shape = tensors[0].shape
    
    if len(shape) < 2:
        return ties_merge_tensor(tensors, keep_ratio)
        
    M = shape[0]
    N = 1
    for s in shape[1:]:
        N *= s
        
    reshaped_tensors = [t.view(M, N).float() for t in tensors]
    
    # Transform to DCT
    DCT_mats = []
    C_M, C_N = None, None
    for X in reshaped_tensors:
        D, C_M, C_N = dct_2d(X)
        DCT_mats.append(D)
        
    # Apply TIES in DCT domain
    merged_dct = ties_merge_tensor(DCT_mats, keep_ratio)
    
    # Inverse DCT
    X_merged = idct_2d(merged_dct, C_M, C_N)
    return X_merged.view(shape).to(tensors[0].dtype)

def evaluate(model, task, device, batch_size=256):
    model = model.to(device)
    model.eval()
    dataset = get_dataset(task, is_train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return (correct / total) * 100.0

def get_adaptive_low_freq_ratio(key, base_ratio=0.1):
    # Map key to its empirical low-frequency energy percentage based on our spectral energy analysis
    if "layer4" in key:
        raw_alpha = 0.1135
    elif "layer3" in key:
        raw_alpha = 0.1474
    elif "layer2" in key:
        raw_alpha = 0.1592
    elif "layer1" in key:
        raw_alpha = 0.2057
    elif "conv1" in key or "bn1" in key:
        raw_alpha = 0.2457
    else:
        return base_ratio # Default for other parameters
        
    # We treat base_ratio as a global scaling multiplier for our empirical low-frequency energy profiles
    return raw_alpha * base_ratio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="wa", choices=["base", "expert", "wa", "ta", "stdfs", "ties", "dare", "s_ties", "s_dare", "lasp", "swsm", "lasp_soft"])
    parser.add_argument("--ta_lambda", type=float, default=0.333, help="Lambda parameter for task arithmetic")
    parser.add_argument("--stdfs_low_freq", type=float, default=0.1, help="Low frequency ratio for STDFS")
    parser.add_argument("--drop_rate", type=float, default=0.2, help="Drop rate for DARE/S-DARE")
    parser.add_argument("--keep_ratio", type=float, default=0.2, help="Keep ratio for TIES/S-TIES")
    parser.add_argument("--gamma", type=float, default=0.05, help="Temperature / smoothness for SWSM")
    parser.add_argument("--head_setting", type=str, default="task_specific", choices=["task_specific", "shared"])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating using device: {device}")
    
    tasks = ["mnist", "fashion", "cifar10"]
    
    # Load base model
    base_model = load_model("checkpoints/base.pt")
    
    if args.method == "base":
        print("Evaluating Base Model...")
        for task in tasks:
            acc = evaluate(base_model, task, device)
            print(f"Task: {task} | Accuracy: {acc:.2f}%")
        return
        
    # Check if expert checkpoints exist
    expert_paths = {task: f"checkpoints/{task}.pt" for task in tasks}
    for task, path in expert_paths.items():
        if not os.path.exists(path):
            print(f"Expert checkpoint not found for {task}: {path}. Please train experts first.")
            return
            
    # Load experts
    experts = {task: load_model(path) for task, path in expert_paths.items()}
    
    if args.method == "expert":
        print("Evaluating Individual Experts...")
        for task in tasks:
            acc = evaluate(experts[task], task, device)
            print(f"Expert {task} on {task}: {acc:.2f}%")
        return
        
    print(f"Merging models using method: {args.method.upper()}")
    print(f"Head Setting: {args.head_setting.upper()}")
    
    # Perform Merge
    merged_state = copy.deepcopy(base_model.state_dict())
    base_state = base_model.state_dict()
    
    # Extract task vectors for backbone
    task_vectors = {task: {} for task in tasks}
    for task in tasks:
        expert_state = experts[task].state_dict()
        for key in base_state.keys():
            # Check if key is part of classification head (fc)
            if args.head_setting == "task_specific" and key.startswith("fc."):
                continue
            task_vectors[task][key] = expert_state[key].float() - base_state[key].float()
            
    # Perform parameter merging
    for key in base_state.keys():
        if args.head_setting == "task_specific" and key.startswith("fc."):
            # We keep fc unchanged in the merged state (we will swap in expert heads at eval time)
            continue
            
        tensors = [task_vectors[task][key] for task in tasks]
        
        if args.method == "wa":
            # Simple averaging of task updates
            merged_update = torch.stack(tensors, dim=0).mean(dim=0)
            merged_state[key] = base_state[key].float() + merged_update
            
        elif args.method == "ta":
            # Task Arithmetic: base + lambda * sum(updates)
            merged_update = torch.stack(tensors, dim=0).sum(dim=0) * args.ta_lambda
            merged_state[key] = base_state[key].float() + merged_update
            
        elif args.method == "stdfs":
            # Spectral Task Decoupling
            merged_update = stdfs_merge_tensor(tensors, args.stdfs_low_freq)
            merged_state[key] = base_state[key].float() + merged_update
            
        elif args.method == "swsm":
            # Spectral Window-based Soft Merging (SWSM)
            merged_update = swsm_merge_tensor(tensors, args.stdfs_low_freq, args.gamma)
            merged_state[key] = base_state[key].float() + merged_update
            
        elif args.method == "lasp":
            # Layer-wise Adaptive Spectral Partitioning (LASP)
            alpha_l = get_adaptive_low_freq_ratio(key, args.stdfs_low_freq)
            merged_update = stdfs_merge_tensor(tensors, alpha_l)
            merged_state[key] = base_state[key].float() + merged_update
            
        elif args.method == "lasp_soft":
            # Layer-wise Adaptive Soft Spectral Partitioning (LASP-Soft)
            alpha_l = get_adaptive_low_freq_ratio(key, args.stdfs_low_freq)
            merged_update = swsm_merge_tensor(tensors, alpha_l, args.gamma)
            merged_state[key] = base_state[key].float() + merged_update
            
        elif args.method == "ties":
            # TIES Merging
            merged_update = ties_merge_tensor(tensors, args.keep_ratio)
            merged_state[key] = base_state[key].float() + merged_update
            
        elif args.method == "dare":
            # DARE Merging
            merged_update = dare_merge_tensor(tensors, args.drop_rate)
            merged_state[key] = base_state[key].float() + merged_update
            
        elif args.method == "s_ties":
            # Spectral TIES Merging
            merged_update = s_ties_merge_tensor(tensors, args.keep_ratio)
            merged_state[key] = base_state[key].float() + merged_update
            
        elif args.method == "s_dare":
            # Spectral DARE Merging
            merged_update = s_dare_merge_tensor(tensors, args.drop_rate)
            merged_state[key] = base_state[key].float() + merged_update
            
    # Prepare merged model
    merged_model = load_model()
    merged_model.load_state_dict(merged_state)
    merged_model = merged_model.to(device)
    
    # Evaluate
    accuracies = {}
    for task in tasks:
        if args.head_setting == "task_specific":
            # Swap in the task-specific classification head from the expert
            eval_model = copy.deepcopy(merged_model)
            eval_model.fc.load_state_dict(experts[task].fc.state_dict())
        else:
            eval_model = merged_model
            
        acc = evaluate(eval_model, task, device)
        accuracies[task] = acc
        print(f"Task: {task} | Accuracy: {acc:.2f}%")
        
    avg_acc = sum(accuracies.values()) / len(accuracies)
    print(f"--- Average Accuracy: {avg_acc:.2f}% ---")

if __name__ == "__main__":
    main()
