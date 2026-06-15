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
    # Generates projection matrices R_k of shape [D, d] mapping local task coordinates to global space
    R = []
    if orthogonal:
        # Non-overlapping coordinate blocks
        for k in range(K):
            R_k = torch.zeros(D, d)
            R_k[k*d : (k+1)*d, :] = torch.eye(d)
            R.append(R_k)
    else:
        # Overlapping (non-orthogonal) subspaces in over-parameterized space
        for k in range(K):
            R_k = torch.randn(D, d)
            # Use QR decomposition to make the columns of each R_k strictly orthonormal
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
            
            # Active task block feature
            eps = torch.randn(d)
            z_t = W[t][c_b] + SIGMAS[t] * eps
            z_t = z_t / torch.norm(z_t) # UNC feature normalization
            
            # Project active task to global space
            z_t_global = torch.matmul(R[t], z_t)
            
            if clean_only:
                z_global = z_t_global
            else:
                # Generate inactive tasks' features and project to global space
                z_inactive_global = torch.zeros(D)
                for k in range(K):
                    if k != t:
                        eps_k = torch.randn(d)
                        z_k = eps_k / torch.norm(eps_k)
                        z_inactive_global += torch.matmul(R[k], z_k)
                # Total global representation is the sum of active and inactive projected vectors
                z_global = z_t_global + z_inactive_global
                
            all_z.append(z_global.unsqueeze(0))
            all_tasks.append(t)
            all_classes.append(c_b)
            
    return torch.cat(all_z), torch.tensor(all_tasks), torch.tensor(all_classes)

def run_experiment(orthogonal=False):
    set_seed(42)
    W = generate_expert_heads()
    R = generate_overlapping_projection_matrices(orthogonal=orthogonal)
    
    # Generate Calibration Data to compute SVD Subspaces (N=100 per task, task-pure)
    calib_z, calib_tasks, _ = generate_overlapping_data(W, R, 100, clean_only=True)
    
    # Compute SVD Subspace Projection Operators P_k = U_k U_k^\top for each task k
    P = []
    U_bases = []
    for k in range(K):
        # Gather calibration samples belonging to task k
        mask = (calib_tasks == k)
        Z_k = calib_z[mask].t() # Shape [D, N_k]
        
        # Compute SVD
        U, S, V = torch.svd(Z_k)
        
        # Keep top d singular vectors
        U_k = U[:, :d]
        U_bases.append(U_k)
        P_k = torch.matmul(U_k, U_k.t())
        P.append(P_k)
        
    # Generate Test Data (with overlapping noise)
    test_z, test_tasks, test_classes = generate_overlapping_data(W, R, 250, clean_only=False)
    
    # Pre-project expert heads into global space: W_k_global = W_k R_k^\top
    W_global = []
    for k in range(K):
        W_k_global = torch.matmul(W[k], R[k].t())
        # Apply Unit-Norm Calibration in global space
        W_k_global = W_k_global / torch.norm(W_k_global, dim=1, keepdim=True)
        W_global.append(W_k_global)
        
    # 1. Evaluate Local PFSR (Upper Bound)
    correct_local = 0
    correct_local_routing = 0
    for b in range(len(test_z)):
        # Evaluate clean local similarities
        u = torch.zeros(K)
        for k in range(K):
            t = test_tasks[b].item()
            if k == t:
                c_b = test_classes[b].item()
                eps = torch.randn(d)
                z_k = W[k][c_b] + SIGMAS[k] * eps
                z_k = z_k / torch.norm(z_k)
            else:
                eps_k = torch.randn(d)
                z_k = eps_k / torch.norm(eps_k)
            
            cos_sims = torch.matmul(W[k], z_k)
            u[k] = torch.max(cos_sims)
            
        calibration_factor = np.sqrt(2 * np.log(10) / d)
        alpha = torch.softmax((u / calibration_factor) / 0.001, dim=0)
        
        if torch.argmax(alpha).item() == test_tasks[b].item():
            correct_local_routing += 1
            
        logits_c = torch.zeros(C)
        for k in range(K):
            t = test_tasks[b].item()
            if k == t:
                c_b = test_classes[b].item()
                eps = torch.randn(d)
                z_k = W[k][c_b] + SIGMAS[k] * eps
                z_k = z_k / torch.norm(z_k)
            else:
                eps_k = torch.randn(d)
                z_k = eps_k / torch.norm(eps_k)
            logits_c += alpha[k] * torch.matmul(W[k], z_k)
            
        if torch.argmax(logits_c).item() == test_classes[b].item():
            correct_local += 1
            
    # 2. Standard Global PFSR (No Projection)
    correct_global = 0
    correct_global_routing = 0
    for b in range(len(test_z)):
        z_b = test_z[b]
        z_norm = torch.norm(z_b)
        u = torch.zeros(K)
        for k in range(K):
            cos_sims = torch.matmul(W_global[k], z_b) / z_norm
            u[k] = torch.max(cos_sims)
            
        calibration_factor = np.sqrt(2 * np.log(10) / d)
        alpha = torch.softmax((u / calibration_factor) / 0.001, dim=0)
        
        if torch.argmax(alpha).item() == test_tasks[b].item():
            correct_global_routing += 1
            
        logits_c = torch.zeros(C)
        for k in range(K):
            expert_logits = torch.matmul(W_global[k], z_b) / z_norm
            logits_c += alpha[k] * expert_logits
            
        if torch.argmax(logits_c).item() == test_classes[b].item():
            correct_global += 1
            
    # 3. SVD-Projected Global PFSR
    correct_svd = 0
    correct_svd_routing = 0
    for b in range(len(test_z)):
        z_b = test_z[b]
        
        u = torch.zeros(K)
        for k in range(K):
            # Project global representation onto the task's own subspace
            z_proj_k = torch.matmul(P[k], z_b)
            z_proj_norm = torch.norm(z_proj_k)
            
            # Compute cosine similarity over the projected representation
            cos_sims = torch.matmul(W_global[k], z_proj_k) / z_proj_norm
            u[k] = torch.max(cos_sims)
            
        calibration_factor = np.sqrt(2 * np.log(10) / d)
        alpha = torch.softmax((u / calibration_factor) / 0.001, dim=0)
        
        if torch.argmax(alpha).item() == test_tasks[b].item():
            correct_svd_routing += 1
            
        logits_c = torch.zeros(C)
        for k in range(K):
            z_proj_k = torch.matmul(P[k], z_b)
            # Use raw unnormalized projected vector for logits to prevent scaling up inactive noise
            expert_logits = torch.matmul(W_global[k], z_proj_k)
            logits_c += alpha[k] * expert_logits
            
        if torch.argmax(logits_c).item() == test_classes[b].item():
            correct_svd += 1
            
    type_str = "Orthogonal (Orthogonal Blocks)" if orthogonal else "Overlapping (Random Subspaces)"
    print(f"--- Configuration: {type_str} ---")
    print(f"Local PFSR Accuracy (Clean Baseline):     {correct_local / len(test_z) * 100.0:.2f}% (Routing: {correct_local_routing / len(test_z) * 100.0:.2f}%)")
    print(f"Standard Global PFSR Accuracy:             {correct_global / len(test_z) * 100.0:.2f}% (Routing: {correct_global_routing / len(test_z) * 100.0:.2f}%)")
    print(f"SVD-Projected Global PFSR Accuracy:        {correct_svd / len(test_z) * 100.0:.2f}% (Routing: {correct_svd_routing / len(test_z) * 100.0:.2f}%)")
    print()

if __name__ == "__main__":
    run_experiment(orthogonal=True)
    run_experiment(orthogonal=False)
