import os
import sys
import math
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

# 1. Model Architecture
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def copy_model_structure(model):
    replica = SimpleMLP()
    replica.load_state_dict(model.state_dict())
    return replica

# 2. Evaluation Helpers
def get_datasets():
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    return test_dataset

def filter_split_dataset(dataset, classes):
    indices = []
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
        
    for idx, target in enumerate(targets):
        if int(target) in classes:
            indices.append(idx)
    return Subset(dataset, indices)

def eval_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total if total > 0 else 0.0

# 3. Euclidean Advanced Merging: DARE and TIES
def merge_dare(models_list, base_model, drop_rate=0.2, scaling_factor=1.0):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    for key in sd_base.keys():
        if sd_base[key].dtype.is_floating_point:
            deltas = [sd_exp[key] - sd_base[key] for sd_exp in sd_experts]
            dare_deltas = []
            for d in deltas:
                mask = (torch.rand_like(d) > drop_rate).float()
                # Rescale kept weights
                d_dare = d * mask / (1.0 - drop_rate + 1e-8)
                dare_deltas.append(d_dare)
            avg_delta = torch.stack(dare_deltas, dim=0).mean(dim=0)
            sd_merged[key] = sd_base[key] + scaling_factor * avg_delta
        else:
            sd_merged[key] = sd_base[key]
    merged.load_state_dict(sd_merged)
    return merged

def merge_ties(models_list, base_model, keep_rate=0.2, scaling_factor=1.0):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    for key in sd_base.keys():
        if sd_base[key].dtype.is_floating_point:
            # Task vectors
            deltas = [sd_exp[key] - sd_base[key] for sd_exp in sd_experts]
            
            # Step 1: Pruning (keep top keep_rate)
            pruned_deltas = []
            for d in deltas:
                if d.numel() == 0:
                    pruned_deltas.append(d)
                    continue
                k = int(keep_rate * d.numel())
                if k == 0:
                    k = 1
                flat_d = d.flatten()
                threshold = torch.topk(torch.abs(flat_d), k).values[-1]
                mask = (torch.abs(d) >= threshold).float()
                pruned_deltas.append(d * mask)
            
            # Step 2: Sign Electing
            stacked = torch.stack(pruned_deltas, dim=0) # [N, ...]
            signs = torch.sign(stacked)
            sum_signs = signs.sum(dim=0)
            consensus_sign = torch.sign(sum_signs)
            
            # Step 3: Disagreement Resolution
            resolved_deltas = []
            for d in pruned_deltas:
                mask = (torch.sign(d) == consensus_sign).float()
                resolved_deltas.append(d * mask)
            
            # Step 4: Average Non-Zero
            resolved_stacked = torch.stack(resolved_deltas, dim=0)
            non_zero_counts = (resolved_stacked != 0).float().sum(dim=0)
            sum_resolved = resolved_stacked.sum(dim=0)
            avg_delta = torch.where(non_zero_counts > 0, sum_resolved / (non_zero_counts + 1e-8), torch.zeros_like(sum_resolved))
            
            sd_merged[key] = sd_base[key] + scaling_factor * avg_delta
        else:
            sd_merged[key] = sd_base[key]
    merged.load_state_dict(sd_merged)
    return merged

# 4. Block-Diagonal Decomposition Helpers
def get_rotation_procrustes(W_k, W_0):
    A = torch.matmul(W_0.t(), W_k)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    R = torch.matmul(U, Vh)
    return R

def cayley_to_skew(R):
    I = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
    Q = torch.linalg.solve(R + I, R - I)
    Q = 0.5 * (Q - Q.t())
    return Q

def cayley_from_skew(Q):
    I = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype)
    R = torch.linalg.solve(I - Q, I + Q)
    return R

def merge_cayley_Q_list(Q_list):
    Q_stack = torch.stack(Q_list, dim=0)
    N = Q_stack.shape[0]
    merged_sum = Q_stack.sum(dim=0)
    norms = torch.norm(Q_stack.view(N, -1), p='fro', dim=1)
    sum_of_norms = norms.sum()
    norm_of_sum = torch.norm(merged_sum, p='fro')
    c = sum_of_norms / (norm_of_sum + 1e-8)
    Q_com = (1.0 / N) * c * merged_sum
    Q_com = 0.5 * (Q_com - Q_com.t())
    return Q_com

def rimo_spectral_balancing(Q_com, t=1.5):
    U, S, Vh = torch.linalg.svd(Q_com, full_matrices=False)
    mean_s = S.mean()
    S_balanced = mean_s + (S - mean_s) / math.sqrt(t)
    Q_balanced = torch.matmul(U * S_balanced.unsqueeze(0), Vh)
    Q_balanced = 0.5 * (Q_balanced - Q_balanced.t())
    return Q_balanced

def block_merge_weight(W_experts, W_0, b=256, t=1.0, res_scale=1.0):
    """
    Perform block-diagonal decomposition and merging of weight matrices.
    If b is larger than matrix dimensions, we use the full matrix.
    W_experts: list of expert weight tensors.
    W_0: base weight tensor.
    """
    out_dim, in_dim = W_0.shape
    device = W_0.device
    dtype = W_0.dtype
    
    # Assert square blocks for simplicity (or handle rectangular by operating on square block-diagonals)
    # We partition into square blocks of size b x b.
    # For simplicity, we split both out_dim and in_dim into steps of size b.
    # Since our MLP hidden layer is 256x256, it perfectly divides by 32, 64, 128, 256.
    
    # We will reconstruct W_merged block by block
    W_merged = torch.zeros_like(W_0)
    total_res_norm = 0.0
    
    num_blocks_row = math.ceil(out_dim / b)
    num_blocks_col = math.ceil(in_dim / b)
    
    for r in range(num_blocks_row):
        r_start, r_end = r * b, min((r + 1) * b, out_dim)
        for c in range(num_blocks_col):
            c_start, c_end = c * b, min((c + 1) * b, in_dim)
            
            # Slice blocks
            W0_block = W_0[r_start:r_end, c_start:c_end]
            W_exp_blocks = [W_k[r_start:r_end, c_start:c_end] for W_k in W_experts]
            
            # If the block is not square, we just do standard task arithmetic or pad it.
            # But in our MLP, all hidden layers are square 256x256, so they are perfectly square.
            # If we are at fc1 (256x784) or fc3 (10x256), they are rectangular.
            # In that case, let's fall back to full SVD or handle square sub-blocks.
            # For simplicity, if block is not square, we do full matrix merging or standard SVD.
            if W0_block.shape[0] != W0_block.shape[1]:
                # Non-square block: fallback to full SVD of this slice
                # Or just standard SVD. Let's do SVD
                # Let's perform full SVD on this rectangular block
                # Since block diagonal is mainly for hidden layers, we can just do full SVD on rectangular layers.
                b_sz = max(W0_block.shape)
                R_list = []
                rho_list = []
                for W_k_b in W_exp_blocks:
                    R_k = get_rotation_procrustes(W_k_b, W0_block)
                    rho_k = W_k_b - torch.matmul(W0_block, R_k)
                    R_list.append(R_k)
                    rho_list.append(rho_k)
                    total_res_norm += torch.norm(rho_k, p='fro').item()
                    
                Q_list = [cayley_to_skew(R) for R in R_list]
                Q_com = merge_cayley_Q_list(Q_list)
                if t > 1.0:
                    Q_com = rimo_spectral_balancing(Q_com, t=t)
                R_merged = cayley_from_skew(Q_com)
                rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
                W_merged[r_start:r_end, c_start:c_end] = torch.matmul(W0_block, R_merged) + res_scale * rho_merged
            else:
                # Square block
                R_list = []
                rho_list = []
                for W_k_b in W_exp_blocks:
                    R_k = get_rotation_procrustes(W_k_b, W0_block)
                    rho_k = W_k_b - torch.matmul(W0_block, R_k)
                    R_list.append(R_k)
                    rho_list.append(rho_k)
                    total_res_norm += torch.norm(rho_k, p='fro').item()
                    
                Q_list = [cayley_to_skew(R) for R in R_list]
                Q_com = merge_cayley_Q_list(Q_list)
                if t > 1.0:
                    Q_com = rimo_spectral_balancing(Q_com, t=t)
                R_merged = cayley_from_skew(Q_com)
                rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
                W_merged[r_start:r_end, c_start:c_end] = torch.matmul(W0_block, R_merged) + res_scale * rho_merged
                
    return W_merged, total_res_norm

def merge_models_block_rimo(models_list, base_model, b=256, t=1.0, res_scale=1.0):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    total_residual_norm = 0.0
    for key in sd_base.keys():
        W_0 = sd_base[key]
        if W_0.dtype.is_floating_point and len(W_0.shape) == 2:
            W_experts = [sd_exp[key] for sd_exp in sd_experts]
            W_merged_block, res_norm = block_merge_weight(W_experts, W_0, b=b, t=t, res_scale=res_scale)
            sd_merged[key] = W_merged_block
            total_residual_norm += res_norm
        else:
            if W_0.dtype.is_floating_point:
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
            else:
                sd_merged[key] = W_0
                
    merged.load_state_dict(sd_merged)
    return merged, total_residual_norm


# 5. Execution Pipeline
def main():
    device = "cpu"
    test_dataset = get_datasets()
    classes_t1 = [0, 1, 2, 3, 4]
    classes_t2 = [5, 6, 7, 8, 9]
    test_t1  = filter_split_dataset(test_dataset, classes_t1)
    test_t2  = filter_split_dataset(test_dataset, classes_t2)
    loader_test_t1  = DataLoader(test_t1, batch_size=128, shuffle=False)
    loader_test_t2  = DataLoader(test_t2, batch_size=128, shuffle=False)
    
    print("--------------------------------------------------")
    print("Evaluating Advanced Baselines and Block Sensitivity")
    print("--------------------------------------------------")
    
    # Load Models
    for regime in ["Standard (Non-OFT)", "Orthogonal Regularized (OFT)"]:
        dir_path = "results" if "Non-OFT" in regime else "results_oft"
        print(f"\n=== Regime: {regime} ===")
        
        base_model = SimpleMLP()
        base_model.load_state_dict(torch.load(f"{dir_path}/base_model.pt", map_location=device))
        
        expert1 = SimpleMLP()
        expert1.load_state_dict(torch.load(f"{dir_path}/expert1.pt", map_location=device))
        
        expert2 = SimpleMLP()
        expert2.load_state_dict(torch.load(f"{dir_path}/expert2.pt", map_location=device))
        
        # SOTA Baselines
        print("\n--- SOTA Euclidean Baselines ---")
        # 1. DARE
        for drop in [0.1, 0.2, 0.5]:
            # DARE averages expert updates
            merged_dare_model = merge_dare([expert1, expert2], base_model, drop_rate=drop, scaling_factor=0.5 if "Non-OFT" in regime else 1.0)
            acc1 = eval_model(merged_dare_model, loader_test_t1, device)
            acc2 = eval_model(merged_dare_model, loader_test_t2, device)
            print(f"  DARE (drop={drop}): Task1={acc1:.4f}, Task2={acc2:.4f}, Avg={0.5*(acc1+acc2):.4f}")
            
        # 2. TIES
        for keep in [0.2, 0.5, 0.8]:
            merged_ties_model = merge_ties([expert1, expert2], base_model, keep_rate=keep, scaling_factor=0.5 if "Non-OFT" in regime else 1.0)
            acc1 = eval_model(merged_ties_model, loader_test_t1, device)
            acc2 = eval_model(merged_ties_model, loader_test_t2, device)
            print(f"  TIES (keep={keep}): Task1={acc1:.4f}, Task2={acc2:.4f}, Avg={0.5*(acc1+acc2):.4f}")
            
        # Block-size Sensitivity (for OrthoMerge/RIMO with t=1.0)
        print("\n--- Block Size Sensitivity (OrthoMerge, res_scale=1.0) ---")
        for b_size in [32, 64, 128, 256]:
            merged_block, res_norm = merge_models_block_rimo([expert1, expert2], base_model, b=b_size, t=1.0, res_scale=1.0)
            acc1 = eval_model(merged_block, loader_test_t1, device)
            acc2 = eval_model(merged_block, loader_test_t2, device)
            print(f"  Block Size b={b_size:3d}: Residual Norm={res_norm:.2f}, Task1={acc1:.4f}, Task2={acc2:.4f}, Avg={0.5*(acc1+acc2):.4f}")
            
    # Measure Latency
    print("\n=== Latency / Execution Time Measurements ===")
    base_model = SimpleMLP()
    base_model.load_state_dict(torch.load("results/base_model.pt", map_location=device))
    expert1 = SimpleMLP()
    expert1.load_state_dict(torch.load("results/expert1.pt", map_location=device))
    expert2 = SimpleMLP()
    expert2.load_state_dict(torch.load("results/expert2.pt", map_location=device))
    
    # 1. Task Arithmetic
    sd_base = base_model.state_dict()
    sd_experts = [expert1.state_dict(), expert2.state_dict()]
    t_start = time.time()
    for _ in range(50):
        for key in sd_base.keys():
            if sd_base[key].dtype.is_floating_point:
                deltas = [sd_exp[key] - sd_base[key] for sd_exp in sd_experts]
                avg_delta = torch.stack(deltas, dim=0).mean(dim=0)
                _ = sd_base[key] + 0.3 * avg_delta
    t_ta = (time.time() - t_start) / 50.0 * 1000 # in ms
    print(f"  Task Arithmetic average execution time: {t_ta:.3f} ms")
    
    # 2. OrthoMerge (b=256)
    t_start = time.time()
    for _ in range(10):
        _, _ = merge_models_block_rimo([expert1, expert2], base_model, b=256, t=1.0, res_scale=1.0)
    t_ortho_256 = (time.time() - t_start) / 10.0 * 1000
    print(f"  OrthoMerge (b=256) average execution time: {t_ortho_256:.3f} ms")
    
    # 3. OrthoMerge (b=32)
    t_start = time.time()
    for _ in range(10):
        _, _ = merge_models_block_rimo([expert1, expert2], base_model, b=32, t=1.0, res_scale=1.0)
    t_ortho_32 = (time.time() - t_start) / 10.0 * 1000
    print(f"  OrthoMerge (b=32) average execution time: {t_ortho_32:.3f} ms")

if __name__ == "__main__":
    main()
