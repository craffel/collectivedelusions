import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# System configurations
L = 14          # number of layers
D = 192         # representation dimension
K = 4           # number of expert tasks (MNIST, FashionMNIST, CIFAR-10, SVHN)
d = D // K      # task block size (48)
C = 10          # number of classes per task
C_tasks = [10, 10, 10, 4] # task-specific class sizes (asymmetric)

task_noises = [0.05, 0.15, 0.45, 0.85]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def generate_orthogonal_matrix(dim, seed):
    set_seed(seed)
    q, r = torch.linalg.qr(torch.randn(dim, dim))
    # Ensure a proper rotation (determinant 1 or just orthogonal)
    d_sign = torch.diag(torch.sign(torch.diag(r)))
    q = torch.mm(q, d_sign)
    return q.to(device)

def generate_rotated_sandbox_data(seed, rho=0.33):
    set_seed(seed)
    
    # Generate orthogonal prototypes
    W_orthogonal = torch.randn(K, C, d)
    Phi_shared = torch.randn(C, d)
    
    C_prototypes = torch.zeros(K, C, d)
    for k in range(K):
        for c in range(C):
            proto = torch.sqrt(torch.tensor(1.0 - rho)) * W_orthogonal[k, c] + torch.sqrt(torch.tensor(rho)) * Phi_shared[c]
            C_prototypes[k, c] = proto / torch.norm(proto)
            
    # Generate task-specific rotation matrices Q_k
    Q_tasks = [generate_orthogonal_matrix(d, seed + 100 * k) for k in range(K)]
    
    # Calibration Split (16 samples per task = 64 total)
    cal_features = []
    cal_labels = []
    cal_tasks = []
    
    for k in range(K):
        for _ in range(16):
            c = torch.randint(0, C_tasks[k], (1,)).item()
            z_base = torch.randn(K, d) * 0.1
            
            # Anisotropic noise (even clean, odd noisy)
            noise_scale = task_noises[k] * torch.tensor([0.1 if j % 2 == 0 else 1.9 for j in range(d)]).to(device)
            raw_noise = noise_scale * torch.randn(d).to(device)
            # Rotate noise using Q_k
            rotated_noise = torch.mv(Q_tasks[k], raw_noise)
            
            z_expert = C_prototypes[k, c].to(device) + rotated_noise
            
            z = torch.zeros(K, d).to(device)
            for j in range(K):
                if j == k:
                    z[j] = z_expert
                else:
                    z[j] = z_base[j].to(device) + torch.randn(d).to(device) * 0.5
                    
            cal_features.append(z.view(-1))
            cal_labels.append(c)
            cal_tasks.append(k)
            
    cal_features = torch.stack(cal_features).to(device)
    cal_labels = torch.tensor(cal_labels, dtype=torch.long).to(device)
    cal_tasks = torch.tensor(cal_tasks, dtype=torch.long).to(device)
    
    # Test Split (250 samples per task = 1000 total)
    test_features = []
    test_labels = []
    test_tasks = []
    
    for k in range(K):
        for _ in range(250):
            c = torch.randint(0, C_tasks[k], (1,)).item()
            z_base = torch.randn(K, d) * 0.1
            
            noise_scale = task_noises[k] * torch.tensor([0.1 if j % 2 == 0 else 1.9 for j in range(d)]).to(device)
            raw_noise = noise_scale * torch.randn(d).to(device)
            rotated_noise = torch.mv(Q_tasks[k], raw_noise)
            
            z_expert = C_prototypes[k, c].to(device) + rotated_noise
            
            z = torch.zeros(K, d).to(device)
            for j in range(K):
                if j == k:
                    z[j] = z_expert
                else:
                    z[j] = z_base[j].to(device) + torch.randn(d).to(device) * 0.5
                    
            test_features.append(z.view(-1))
            test_labels.append(c)
            test_tasks.append(k)
            
    test_features = torch.stack(test_features).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
    test_tasks = torch.tensor(test_tasks, dtype=torch.long).to(device)
    
    C_prototypes = C_prototypes.to(device)
    return C_prototypes, cal_features, cal_labels, cal_tasks, test_features, test_labels, test_tasks, Q_tasks

def estimate_diagonal_fisher(C_prototypes, cal_features, cal_labels, cal_tasks, Q_tasks=None, rotate_back=False):
    F = torch.zeros(K, C, d).to(device)
    for k in range(K):
        mask = (cal_tasks == k)
        feat_k = cal_features[mask]
        z_k = feat_k.view(-1, K, d)[:, k, :] # [16, d]
        
        if rotate_back and Q_tasks is not None:
            # Rotate back to make it axis-aligned
            # z_k_aligned = z_k * Q_k
            # Since Q_k is orthogonal, z_k_aligned = z_k \times Q_k
            Q_k = Q_tasks[k]
            z_k = torch.mm(z_k, Q_k) # [16, d]
            
        # Compute pooled within-class coordinate variance to isolate noise and prevent centroid conflation
        labels_k = cal_labels[mask]
        sq_deviations = torch.zeros(d).to(device)
        df = 0  # degrees of freedom
        for c in range(C_tasks[k]):
            class_mask = (labels_k == c)
            class_samples = z_k[class_mask]
            Nc = len(class_samples)
            if Nc > 1:
                class_mean = class_samples.mean(dim=0)
                sq_deviations += torch.sum((class_samples - class_mean) ** 2, dim=0)
                df += (Nc - 1)

        if df > 0:
            var_k = sq_deviations / df + 1e-5
        else:
            var_k = torch.var(z_k, dim=0, unbiased=True) + 1e-5

        F_k_raw = 1.0 / var_k
        beta = 0.5
        gamma = 0.7
        smoothed = (F_k_raw + beta) ** gamma
        F_k_norm = smoothed / torch.sum(smoothed)
        for c in range(C_tasks[k]):
            F[k, c] = F_k_norm
    return F

def fisher_weighted_cosine_similarity(z_k, W_k, F_k):
    z_k_expanded = z_k.unsqueeze(1)
    W_k_expanded = W_k.unsqueeze(0)
    F_k_expanded = F_k.unsqueeze(0)
    num = torch.sum(F_k_expanded * W_k_expanded * z_k_expanded, dim=-1)
    den1 = torch.sqrt(torch.sum(F_k_expanded * (W_k_expanded ** 2), dim=-1))
    den2 = torch.sqrt(torch.sum(F_k_expanded * (z_k_expanded ** 2), dim=-1))
    sim = num / (den1 * den2 + 1e-8)
    return sim

def standard_cosine_similarity(z_k, W_k):
    z_k_norm = z_k / (torch.norm(z_k, dim=-1, keepdim=True) + 1e-8)
    W_k_norm = W_k / (torch.norm(W_k, dim=-1, keepdim=True) + 1e-8)
    sim = torch.mm(z_k_norm, W_k_norm.t())
    return sim

def compute_merged_logits(z, alpha, C_prototypes, tasks, Q_tasks=None):
    B_size = len(z)
    logits = torch.zeros(B_size, C).to(device)
    z_blocks = z.view(B_size, K, d)
    for b in range(B_size):
        k = tasks[b].item()
        z_base_k = torch.randn(d).to(device) * 0.1
        z_expert_k = z_blocks[b, k]
        interference = torch.zeros(d).to(device)
        for i in range(K):
            if i != k:
                interference += alpha[b, i] * (torch.randn(d).to(device) * 0.5)
        z_merged_k = alpha[b, k] * z_expert_k + (1.0 - alpha[b, k]) * z_base_k + interference
        W_k = C_prototypes[k]
        b_logits = torch.mv(W_k, z_merged_k)
        if C_tasks[k] < C:
            b_logits[C_tasks[k]:] = -1e9
        logits[b] = b_logits
    return logits

# Compute CC-CSC denom dynamically
def get_cc_csc_denom(C_prototypes):
    cc_denom = []
    for k in range(K):
        W_k = C_prototypes[k, :C_tasks[k]] # [C_tasks[k], d]
        W_k_norm = W_k / (torch.norm(W_k, dim=-1, keepdim=True) + 1e-8)
        R_k = torch.mm(W_k_norm, W_k_norm.t())
        frob_sq = torch.sum(R_k ** 2)
        C_k = C_tasks[k]
        d_eff_k = (C_k ** 2) / frob_sq
        d_eff_k = torch.clamp(d_eff_k, min=1.0, max=float(d))
        denom_k = torch.sqrt(2 * np.log(C_k) / d_eff_k)
        cc_denom.append(denom_k)
    return torch.stack(cc_denom).to(device)

def get_mbh_coefficients(method_name, feat, C_prototypes, Fisher_M, Q_tasks=None):
    B_size = len(feat)
    z_blocks = feat.view(B_size, K, d)
    u = torch.zeros(B_size, K).to(device)
    
    # CC-CSC denom
    csc_denom = get_cc_csc_denom(C_prototypes)
    
    for k in range(K):
        W_k = C_prototypes[k, :C_tasks[k]]
        z_k = z_blocks[:, k, :]
        
        if method_name == "FIOSR-Rotated" and Q_tasks is not None:
            # Rotate back both features and class prototypes to align with Fisher
            Q_k = Q_tasks[k]
            z_k = torch.mm(z_k, Q_k) # [B, d]
            W_k = torch.mm(W_k, Q_k) # [C, d]
            F_k = Fisher_M[k, :C_tasks[k]]
            sims = fisher_weighted_cosine_similarity(z_k, W_k, F_k)
        elif method_name == "FIOSR-Diag":
            # Apply diagonal Fisher directly on rotated coordinates (ignores correlation)
            F_k = Fisher_M[k, :C_tasks[k]]
            sims = fisher_weighted_cosine_similarity(z_k, W_k, F_k)
        else: # PFSR_MBH (standard Cosine)
            sims = standard_cosine_similarity(z_k, W_k)
            
        u[:, k] = torch.max(sims, dim=-1)[0]
        
    u_calibrated = u / csc_denom
    tau = 0.001
    alpha_raw = torch.softmax(u_calibrated / tau, dim=-1)
    dominant_tasks = torch.argmax(u_calibrated, dim=-1)
    alpha_mbh = alpha_raw.clone()
    for k in range(K):
        mask = (dominant_tasks == k)
        if torch.sum(mask) > 0:
            mean_alpha = torch.mean(alpha_raw[mask], dim=0)
            alpha_mbh[mask] = mean_alpha
    return alpha_mbh

def evaluate_method(method_name, C_prototypes, Fisher_M, test_features, test_labels, test_tasks, batch_size, Q_tasks=None):
    total_correct = 0
    total_samples = len(test_features)
    
    task_accuracies = []
    for k in range(K):
        mask = (test_tasks == k)
        feat_k = test_features[mask]
        lbl_k = test_labels[mask]
        tasks_k = test_tasks[mask]
        num_task_samples = len(feat_k)
        correct_k = 0
        for i in range(0, num_task_samples, batch_size):
            end_idx = min(i + batch_size, num_task_samples)
            batch_feat = feat_k[i:end_idx]
            batch_lbl = lbl_k[i:end_idx]
            batch_tasks = tasks_k[i:end_idx]
            alpha = get_mbh_coefficients(method_name, batch_feat, C_prototypes, Fisher_M, Q_tasks)
            pred_logits = compute_merged_logits(batch_feat, alpha, C_prototypes, batch_tasks, Q_tasks)
            preds = torch.argmax(pred_logits, dim=-1)
            correct_k += torch.sum(preds == batch_lbl).item()
        task_accuracies.append(correct_k / num_task_samples)
    return np.mean(task_accuracies) * 100.0

seeds = list(range(42, 52))

pfsr_results = []
fiosr_diag_results = []
fiosr_rot_results = []

print("Running Rotated Noise Experiment over 10 seeds...")
for seed in seeds:
    C_prototypes, cal_f, cal_l, cal_t, test_f, test_l, test_t, Q_tasks = generate_rotated_sandbox_data(seed, rho=0.33)
    mean_cal = cal_f.mean(dim=0, keepdim=True)
    cal_f = cal_f - mean_cal
    test_f = test_f - mean_cal
    
    # 1. Standard Flat Cosine
    acc_pfsr = evaluate_method("PFSR_MBH", C_prototypes, None, test_f, test_l, test_t, batch_size=256, Q_tasks=Q_tasks)
    pfsr_results.append(acc_pfsr)
    
    # 2. Diagonal Fisher on Rotated Data (ignores correlation, suffers from isotropic degradation)
    Fisher_Diag = estimate_diagonal_fisher(C_prototypes, cal_f, cal_l, cal_t, Q_tasks=None, rotate_back=False)
    acc_fiosr_diag = evaluate_method("FIOSR-Diag", C_prototypes, Fisher_Diag, test_f, test_l, test_t, batch_size=256, Q_tasks=Q_tasks)
    fiosr_diag_results.append(acc_fiosr_diag)
    
    # 3. Manifold-Aligned/K-FAC Block-Diagonal Fisher (rotates back to align with primary Fisher axes)
    Fisher_Rot = estimate_diagonal_fisher(C_prototypes, cal_f, cal_l, cal_t, Q_tasks=Q_tasks, rotate_back=True)
    acc_fiosr_rot = evaluate_method("FIOSR-Rotated", C_prototypes, Fisher_Rot, test_f, test_l, test_t, batch_size=256, Q_tasks=Q_tasks)
    fiosr_rot_results.append(acc_fiosr_rot)
    
    print(f"Seed {seed:<2} | PFSR+MBH: {acc_pfsr:.2f}% | FIOSR-Diag: {acc_fiosr_diag:.2f}% | FIOSR-Rotated (K-FAC/Block): {acc_fiosr_rot:.2f}%")

print("\n--- FINAL ROTATED NOISE EXPERIMENT RESULTS ---")
print(f"PFSR + MBH (Flat Cosine):               {np.mean(pfsr_results):.2f}% +/- {np.std(pfsr_results):.2f}%")
print(f"FIOSR-Diag (Diagonal Fisher):           {np.mean(fiosr_diag_results):.2f}% +/- {np.std(fiosr_diag_results):.2f}%")
print(f"FIOSR-Rotated (K-FAC / Block-Diagonal): {np.mean(fiosr_rot_results):.2f}% +/- {np.std(fiosr_rot_results):.2f}%")
