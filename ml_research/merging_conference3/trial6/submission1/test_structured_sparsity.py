# Evaluate Structured vs Unstructured Sparsity for Residual-EHPB
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

D = 192
K = 4
num_classes = 10

# Generate task vectors for 14 layers
L = 14
V_task_layers = {k: [] for k in range(K)}
for l in range(L - 1):
    for k in range(K):
        V_k = torch.randn(D, D) / math.sqrt(D) * 0.45
        V_task_layers[k].append(V_k)

# Evaluate weight reconstruction error under unstructured vs structured masks
p_sparsity = 0.05 # 5% sparsity

print(f"--- Residual Masking Analysis (Sparsity Budget: {p_sparsity*100}%) ---")

unstructured_errors = []
structured_row_errors = []

for l in range(L - 1):
    # Sum of absolute task vectors for this layer
    task_sum = torch.zeros(D, D)
    for k in range(K):
        task_sum += torch.abs(V_task_layers[k][l])
        
    # 1. Unstructured Mask (top 5% individual elements)
    threshold_unstructured = torch.quantile(task_sum, 1.0 - p_sparsity)
    mask_unstruct = (task_sum >= threshold_unstructured).float()
    
    # 2. Structured Mask (top 5% rows by L1 norm)
    row_norms = task_sum.sum(dim=1) # [D]
    threshold_row = torch.quantile(row_norms, 1.0 - p_sparsity)
    mask_row_vector = (row_norms >= threshold_row).float()
    mask_struct_row = mask_row_vector.unsqueeze(1).expand(D, D)
    
    # Keys
    keys = []
    for k in range(K):
        r_k = torch.randn(D).sign()
        c_k = torch.randn(D).sign()
        keys.append(torch.outer(r_k, c_k))
    keys = torch.stack(keys) # [K, D, D]
    
    # Superposition
    W_holo_unstruct = torch.zeros(D, D)
    W_holo_struct_row = torch.zeros(D, D)
    
    V_stacked = torch.stack([V_task_layers[k][l] for k in range(K)])
    
    W_holo_unstruct = torch.sum(V_stacked * (1.0 - mask_unstruct).unsqueeze(0) * keys, dim=0)
    W_holo_struct_row = torch.sum(V_stacked * (1.0 - mask_struct_row).unsqueeze(0) * keys, dim=0)
    
    # Compute relative reconstruction error
    errors_unstruct = []
    errors_struct_row = []
    
    for k in range(K):
        W_reconstructed_unstruct = W_holo_unstruct * keys[k] + V_task_layers[k][l] * mask_unstruct
        err_unstruct = torch.norm(W_reconstructed_unstruct - V_task_layers[k][l]) / torch.norm(V_task_layers[k][l])
        errors_unstruct.append(err_unstruct.item())
        
        W_reconstructed_struct_row = W_holo_struct_row * keys[k] + V_task_layers[k][l] * mask_struct_row
        err_struct_row = torch.norm(W_reconstructed_struct_row - V_task_layers[k][l]) / torch.norm(V_task_layers[k][l])
        errors_struct_row.append(err_struct_row.item())
        
    unstructured_errors.append(np.mean(errors_unstruct))
    structured_row_errors.append(np.mean(errors_struct_row))

print(f"Unstructured Residual EHPB Relative Error: {np.mean(unstructured_errors)*100:.2f}%")
print(f"Structured Row-wise Residual EHPB Relative Error: {np.mean(structured_row_errors)*100:.2f}%")
