# Evaluate EHPB on Correlated Low-Rank PEFT (LoRA) Manifolds
import math
import torch
import numpy as np

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

D = 192
K = 4
r_lora = 8 # LoRA Rank

print("Sweeping task correlation factor rho...")

# We will sweep the correlation factor rho which represents the shared subspace ratio
for rho in [0.0, 0.25, 0.50, 0.75, 0.95]:
    # Base/Shared LoRA components
    A_shared = torch.randn(D, r_lora) / math.sqrt(D)
    B_shared = torch.randn(r_lora, D) / math.sqrt(D)
    
    V_tasks = []
    for k in range(K):
        # Each task's LoRA adapter is a blend of shared and task-specific components
        A_task = torch.randn(D, r_lora) / math.sqrt(D)
        B_task = torch.randn(r_lora, D) / math.sqrt(D)
        
        # Linear interpolation with correlation factor rho
        A_k = math.sqrt(rho) * A_shared + math.sqrt(1.0 - rho) * A_task
        B_k = math.sqrt(rho) * B_shared + math.sqrt(1.0 - rho) * B_task
        
        V_k = torch.matmul(A_k, B_k) * 0.45
        V_tasks.append(V_k)
        
    V_tasks = torch.stack(V_tasks) # [K, D, D]
    
    # Key generation
    keys = []
    for k in range(K):
        r_k = torch.randn(D).sign()
        c_k = torch.randn(D).sign()
        keys.append(torch.outer(r_k, c_k))
    keys = torch.stack(keys) # [K, D, D]
    
    # Holographic Superposition
    W_holo = torch.sum(V_tasks * keys, dim=0)
    
    # Compute relative reconstruction error for each task
    errors = []
    for k in range(K):
        # Demodulate
        W_reconstructed_k = W_holo * keys[k]
        # Relative error
        error = torch.norm(W_reconstructed_k - V_tasks[k]) / (torch.norm(V_tasks[k]) + 1e-8)
        errors.append(error.item())
        
    mean_err = np.mean(errors) * 100
    print(f"Correlation rho = {rho:.2f} | EHPB Mean Relative Weight Reconstruction Error: {mean_err:.2f}%")
