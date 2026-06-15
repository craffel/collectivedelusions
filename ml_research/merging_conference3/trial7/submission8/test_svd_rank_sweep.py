import torch
import torch.nn as nn
import numpy as np
from simulate import generate_expert_heads, set_seed

D = 1024  # Real-world over-parameterized representation dimension
K = 4
d = 48    # Task-intrinsic dimension
C = 10
SIGMAS = [0.05, 0.05, 0.35, 1.25]

def generate_overlapping_projection_matrices(orthogonal=False):
    R = []
    if orthogonal:
        for k in range(K):
            R_k = torch.zeros(D, d)
            R_k[k*d : (k+1)*d, :] = torch.eye(d)
            R.append(R_k)
    else:
        for k in range(K):
            R_k = torch.randn(D, d)
            Q, _ = torch.linalg.qr(R_k)
            R.append(Q)
    return R

def generate_overlapping_data(W, R, num_samples_per_task, clean_only=False):
    all_z = []
    all_tasks = []
    all_classes = []
    
    for t in range(K):
        for _ in range(num_samples_per_task):
            c_b = np.random.randint(0, C)
            eps = torch.randn(d)
            z_t = W[t][c_b] + SIGMAS[t] * eps
            z_t = z_t / torch.norm(z_t)
            z_t_global = torch.matmul(R[t], z_t)
            
            if clean_only:
                z_global = z_t_global
            else:
                z_inactive_global = torch.zeros(D)
                for k in range(K):
                    if k != t:
                        eps_k = torch.randn(d)
                        z_k = eps_k / torch.norm(eps_k)
                        z_inactive_global += torch.matmul(R[k], z_k)
                z_global = z_t_global + z_inactive_global
                
            all_z.append(z_global.unsqueeze(0))
            all_tasks.append(t)
            all_classes.append(c_b)
            
    return torch.cat(all_z), torch.tensor(all_tasks), torch.tensor(all_classes)

def run_experiment_rank_sweep():
    set_seed(42)
    W = generate_expert_heads()
    R = generate_overlapping_projection_matrices(orthogonal=False)
    
    # Calibration data
    calib_z, calib_tasks, _ = generate_overlapping_data(W, R, 100, clean_only=True)
    # Test data
    test_z, test_tasks, test_classes = generate_overlapping_data(W, R, 250, clean_only=False)
    
    # Pre-project expert heads
    W_global = []
    for k in range(K):
        W_k_global = torch.matmul(W[k], R[k].t())
        W_k_global = W_k_global / torch.norm(W_k_global, dim=1, keepdim=True)
        W_global.append(W_k_global)
        
    # SVD for calibration
    U_all = []
    for k in range(K):
        mask = (calib_tasks == k)
        Z_k = calib_z[mask].t() # Shape [D, N_k]
        U, S, V = torch.svd(Z_k)
        U_all.append(U)
        
    ranks_to_sweep = [8, 16, 24, 32, 48, 64, 96, 128, 256]
    print(f"Sweep results for rank r in overlapping subspace projections:")
    print(f"{'Rank r':<8}{'Joint Acc (%)':<15}{'Routing Acc (%)':<18}{'Storage (KB/expert)':<22}")
    
    for r in ranks_to_sweep:
        P = []
        for k in range(K):
            U_k = U_all[k][:, :r]
            P_k = torch.matmul(U_k, U_k.t())
            P.append(P_k)
            
        correct_svd = 0
        correct_svd_routing = 0
        for b in range(len(test_z)):
            z_b = test_z[b]
            u = torch.zeros(K)
            for k in range(K):
                z_proj_k = torch.matmul(P[k], z_b)
                z_proj_norm = torch.norm(z_proj_k)
                cos_sims = torch.matmul(W_global[k], z_proj_k) / z_proj_norm
                u[k] = torch.max(cos_sims)
                
            calibration_factor = np.sqrt(2 * np.log(10) / d)
            alpha = torch.softmax((u / calibration_factor) / 0.001, dim=0)
            
            if torch.argmax(alpha).item() == test_tasks[b].item():
                correct_svd_routing += 1
                
            logits_c = torch.zeros(C)
            for k in range(K):
                z_proj_k = torch.matmul(P[k], z_b)
                expert_logits = torch.matmul(W_global[k], z_proj_k)
                logits_c += alpha[k] * expert_logits
                
            if torch.argmax(logits_c).item() == test_classes[b].item():
                correct_svd += 1
                
        joint_acc = correct_svd / len(test_z) * 100.0
        routing_acc = correct_svd_routing / len(test_z) * 100.0
        # Storage in KB: each parameter is 4 bytes (float32). We store U_k of shape [D, r]. D = 1024.
        storage_kb = (D * r * 4) / 1024.0
        
        print(f"{r:<8}{joint_acc:<15.2f}{routing_acc:<18.2f}{storage_kb:<22.2f}")

if __name__ == "__main__":
    run_experiment_rank_sweep()
