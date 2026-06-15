import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def generate_sandbox_data(seed, N_cal_per_task, rho=0.33):
    set_seed(seed)
    W_orthogonal = torch.randn(K, C, d)
    Phi_shared = torch.randn(C, d)
    
    C_prototypes = torch.zeros(K, C, d)
    for k in range(K):
        for c in range(C):
            proto = torch.sqrt(torch.tensor(1.0 - rho)) * W_orthogonal[k, c] + torch.sqrt(torch.tensor(rho)) * Phi_shared[c]
            C_prototypes[k, c] = proto / torch.norm(proto)
            
    cal_features = []
    cal_labels = []
    cal_tasks = []
    
    for k in range(K):
        for _ in range(N_cal_per_task):
            c = torch.randint(0, C_tasks[k], (1,)).item()
            z_base = torch.randn(K, d) * 0.1
            noise_scale = task_noises[k] * torch.tensor([0.1 if j % 2 == 0 else 1.9 for j in range(d)])
            z_expert = C_prototypes[k, c] + noise_scale * torch.randn(d)
            
            z = torch.zeros(K, d)
            for j in range(K):
                if j == k:
                    z[j] = z_expert
                else:
                    z[j] = z_base[j] + torch.randn(d) * 0.5
                    
            cal_features.append(z.view(-1))
            cal_labels.append(c)
            cal_tasks.append(k)
            
    cal_features = torch.stack(cal_features).to(device)
    cal_labels = torch.tensor(cal_labels, dtype=torch.long).to(device)
    cal_tasks = torch.tensor(cal_tasks, dtype=torch.long).to(device)
    
    test_features = []
    test_labels = []
    test_tasks = []
    
    for k in range(K):
        for _ in range(250):
            c = torch.randint(0, C_tasks[k], (1,)).item()
            z_base = torch.randn(K, d) * 0.1
            noise_scale = task_noises[k] * torch.tensor([0.1 if j % 2 == 0 else 1.9 for j in range(d)])
            z_expert = C_prototypes[k, c] + noise_scale * torch.randn(d)
            
            z = torch.zeros(K, d)
            for j in range(K):
                if j == k:
                    z[j] = z_expert
                else:
                    z[j] = z_base[j] + torch.randn(d) * 0.5
                    
            test_features.append(z.view(-1))
            test_labels.append(c)
            test_tasks.append(k)
            
    test_features = torch.stack(test_features).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
    test_tasks = torch.tensor(test_tasks, dtype=torch.long).to(device)
    
    C_prototypes = C_prototypes.to(device)
    return C_prototypes, cal_features, cal_labels, cal_tasks, test_features, test_labels, test_tasks

def estimate_diagonal_fisher(C_prototypes, cal_features, cal_labels, cal_tasks):
    F = torch.zeros(K, C, d).to(device)
    for k in range(K):
        mask = (cal_tasks == k)
        feat_k = cal_features[mask]
        z_k = feat_k.view(-1, K, d)[:, k, :]
        
        labels_k = cal_labels[mask]
        sq_deviations = torch.zeros(d).to(device)
        df = 0
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

csc_denom = torch.tensor([np.sqrt(2 * np.log(C_tasks[k]) / d) for k in range(K)]).to(device)

def compute_merged_logits(z, alpha, C_prototypes, tasks):
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

def evaluate_pfsr_and_fiosr(C_prototypes, cal_features, cal_labels, cal_tasks, test_features, test_labels, test_tasks):
    # 1. Estimate Fisher
    F = estimate_diagonal_fisher(C_prototypes, cal_features, cal_labels, cal_tasks)
    
    # 2. PFSR Routing (flat Cosine)
    test_blocks = test_features.view(-1, K, d)
    pfsr_scores = torch.zeros(len(test_features), K).to(device)
    for k in range(K):
        W_k = C_prototypes[k]
        sims_k = standard_cosine_similarity(test_blocks[:, k, :], W_k)
        max_sims_k, _ = torch.max(sims_k, dim=-1)
        pfsr_scores[:, k] = max_sims_k / csc_denom[k]
        
    pfsr_alpha = torch.softmax(pfsr_scores, dim=-1)
    pfsr_logits = compute_merged_logits(test_features, pfsr_alpha, C_prototypes, test_tasks)
    pfsr_preds = torch.argmax(pfsr_logits, dim=-1)
    pfsr_acc = (pfsr_preds == test_labels).float().mean().item() * 100.0
    
    # 3. FIOSR Routing (Fisher-weighted Cosine)
    fiosr_scores = torch.zeros(len(test_features), K).to(device)
    for k in range(K):
        W_k = C_prototypes[k]
        F_k = F[k]
        sims_k = fisher_weighted_cosine_similarity(test_blocks[:, k, :], W_k, F_k)
        max_sims_k, _ = torch.max(sims_k, dim=-1)
        fiosr_scores[:, k] = max_sims_k / csc_denom[k]
        
    fiosr_alpha = torch.softmax(fiosr_scores, dim=-1)
    fiosr_logits = compute_merged_logits(test_features, fiosr_alpha, C_prototypes, test_tasks)
    fiosr_preds = torch.argmax(fiosr_logits, dim=-1)
    fiosr_acc = (fiosr_preds == test_labels).float().mean().item() * 100.0
    
    return pfsr_acc, fiosr_acc

def main():
    seeds = list(range(42, 52))
    N_cal_list = [2, 4, 8, 16, 32, 64, 128]
    
    print("Starting Calibration Sensitivity Sweep...")
    print(f"Sweeping N_cal_per_task: {N_cal_list} over {len(seeds)} seeds...")
    
    results = {}
    for N_cal in N_cal_list:
        pfsr_accs = []
        fiosr_accs = []
        for seed in seeds:
            C_prototypes, cal_features, cal_labels, cal_tasks, test_features, test_labels, test_tasks = generate_sandbox_data(seed, N_cal)
            mean_cal = cal_features.mean(dim=0, keepdim=True)
            cal_features = cal_features - mean_cal
            test_features = test_features - mean_cal
            pfsr_acc, fiosr_acc = evaluate_pfsr_and_fiosr(C_prototypes, cal_features, cal_labels, cal_tasks, test_features, test_labels, test_tasks)
            pfsr_accs.append(pfsr_acc)
            fiosr_accs.append(fiosr_acc)
            
        pfsr_mean = np.mean(pfsr_accs)
        pfsr_std = np.std(pfsr_accs)
        fiosr_mean = np.mean(fiosr_accs)
        fiosr_std = np.std(fiosr_accs)
        
        results[N_cal] = {
            "pfsr_mean": pfsr_mean,
            "pfsr_std": pfsr_std,
            "fiosr_mean": fiosr_mean,
            "fiosr_std": fiosr_std
        }
        print(f"N_cal={N_cal} (Total={N_cal*K}): PFSR = {pfsr_mean:.2f}% +- {pfsr_std:.2f}%, FIOSR = {fiosr_mean:.2f}% +- {fiosr_std:.2f}%")
        
    print("\n--- Summary Markdown Table ---")
    print("| Calibration Size $N_c$ (per task) | Total Calibration Size | PFSR (Flat Cosine) Accuracy (%) | FIOSR (Ours) Accuracy (%) | Absolute Gain (%) |")
    print("|---|---|---|---|---|")
    for N_cal in N_cal_list:
        p_m, p_s = results[N_cal]["pfsr_mean"], results[N_cal]["pfsr_std"]
        f_m, f_s = results[N_cal]["fiosr_mean"], results[N_cal]["fiosr_std"]
        gain = f_m - p_m
        print(f"| {N_cal} | {N_cal*K} | {p_m:.2f}\\% $\\pm$ {p_s:.2f}\\% | {f_m:.2f}\\% $\\pm$ {f_s:.2f}\\% | +{gain:.2f}\\% |")

if __name__ == "__main__":
    main()
