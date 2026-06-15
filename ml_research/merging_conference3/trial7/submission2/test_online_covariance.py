import torch
import torch.nn as nn
import numpy as np

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
    d_sign = torch.diag(torch.sign(torch.diag(r)))
    q = torch.mm(q, d_sign)
    return q.to(device)

def generate_rotated_sandbox_data(seed, rho=0.33):
    set_seed(seed)
    W_orthogonal = torch.randn(K, C, d)
    Phi_shared = torch.randn(C, d)
    
    C_prototypes = torch.zeros(K, C, d)
    for k in range(K):
        for c in range(C):
            proto = torch.sqrt(torch.tensor(1.0 - rho)) * W_orthogonal[k, c] + torch.sqrt(torch.tensor(rho)) * Phi_shared[c]
            C_prototypes[k, c] = proto / torch.norm(proto)
            
    Q_tasks = [generate_orthogonal_matrix(d, seed + 100 * k) for k in range(K)]
    
    cal_features = []
    cal_labels = []
    cal_tasks = []
    
    for k in range(K):
        for _ in range(16):
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
    
    return C_prototypes.to(device), cal_features, cal_labels, cal_tasks, test_features, test_labels, test_tasks, Q_tasks

def estimate_covariance(z_k, labels_k, C_k, alpha=0.5):
    d_dim = z_k.size(1)
    sq_deviations = torch.zeros(d_dim, d_dim).to(device)
    df = 0
    for c in range(C_k):
        class_mask = (labels_k == c)
        class_samples = z_k[class_mask]
        Nc = len(class_samples)
        if Nc > 1:
            class_mean = class_samples.mean(dim=0)
            diff = class_samples - class_mean
            sq_deviations += torch.mm(diff.t(), diff)
            df += (Nc - 1)
    if df > 0:
        Sigma_empirical = sq_deviations / df
    else:
        z_mean = z_k.mean(dim=0)
        diff = z_k - z_mean
        Sigma_empirical = torch.mm(diff.t(), diff) / (len(z_k) - 1)
        
    scale = torch.trace(Sigma_empirical) / d_dim
    Sigma = (1.0 - alpha) * Sigma_empirical + alpha * torch.eye(d_dim).to(device) * scale
    return Sigma

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

def get_cc_csc_denom(C_prototypes):
    cc_denom = []
    for k in range(K):
        W_k = C_prototypes[k, :C_tasks[k]]
        W_k_norm = W_k / (torch.norm(W_k, dim=-1, keepdim=True) + 1e-8)
        R_k = torch.mm(W_k_norm, W_k_norm.t())
        frob_sq = torch.sum(R_k ** 2)
        C_k = C_tasks[k]
        d_eff_k = (C_k ** 2) / frob_sq
        d_eff_k = torch.clamp(d_eff_k, min=1.0, max=float(d))
        denom_k = torch.sqrt(2 * np.log(C_k) / d_eff_k)
        cc_denom.append(denom_k)
    return torch.stack(cc_denom).to(device)

def get_mbh_coefficients_online(method_name, feat, C_prototypes, Fisher_M, U_est=None, Q_tasks=None):
    B_size = len(feat)
    z_blocks = feat.view(B_size, K, d)
    u = torch.zeros(B_size, K).to(device)
    
    csc_denom = get_cc_csc_denom(C_prototypes)
    
    for k in range(K):
        W_k = C_prototypes[k, :C_tasks[k]]
        z_k = z_blocks[:, k, :]
        
        if method_name == "FIOSR-Rotated" and Q_tasks is not None:
            Q_k = Q_tasks[k]
            z_k = torch.mm(z_k, Q_k)
            W_k = torch.mm(W_k, Q_k)
            F_k = Fisher_M[k, :C_tasks[k]]
            sims = fisher_weighted_cosine_similarity(z_k, W_k, F_k)
        elif method_name == "FIOSR-Online" and U_est is not None:
            U_k = U_est[k]
            z_k = torch.mm(z_k, U_k)
            W_k = torch.mm(W_k, U_k)
            F_k = Fisher_M[k, :C_tasks[k]]
            sims = fisher_weighted_cosine_similarity(z_k, W_k, F_k)
        elif method_name == "FIOSR-Diag":
            F_k = Fisher_M[k, :C_tasks[k]]
            sims = fisher_weighted_cosine_similarity(z_k, W_k, F_k)
        else:
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

def evaluate_method_online(method_name, C_prototypes, Fisher_M, test_features, test_labels, test_tasks, batch_size, U_est=None, Q_tasks=None, seed=42):
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
            
            # Reset random seed right before evaluation of the batch to make it fully deterministic
            set_seed(seed + 1000 * k + i)
            
            alpha = get_mbh_coefficients_online(method_name, batch_feat, C_prototypes, Fisher_M, U_est, Q_tasks)
            pred_logits = compute_merged_logits(batch_feat, alpha, C_prototypes, batch_tasks, Q_tasks)
            preds = torch.argmax(pred_logits, dim=-1)
            correct_k += torch.sum(preds == batch_lbl).item()
        task_accuracies.append(correct_k / num_task_samples)
    return np.mean(task_accuracies) * 100.0

seeds = list(range(42, 52))

pfsr_results = []
fiosr_diag_results = []
fiosr_rot_results = []
fiosr_online_results = []

print("Running Rotated Noise Experiment with Deterministic Online Covariance Alignment...")
for seed in seeds:
    C_prototypes, cal_f, cal_l, cal_t, test_f, test_l, test_t, Q_tasks = generate_rotated_sandbox_data(seed, rho=0.33)
    mean_cal = cal_f.mean(dim=0, keepdim=True)
    cal_f = cal_f - mean_cal
    test_f = test_f - mean_cal
    
    # 1. Standard Flat Cosine
    acc_pfsr = evaluate_method_online("PFSR_MBH", C_prototypes, None, test_f, test_l, test_t, batch_size=256, Q_tasks=Q_tasks, seed=seed)
    pfsr_results.append(acc_pfsr)
    
    # 2. Diagonal Fisher directly on Rotated space
    df_diag = torch.zeros(K, C, d).to(device)
    for k in range(K):
        mask = (cal_t == k)
        z_k = cal_f[mask].view(-1, K, d)[:, k, :]
        var_k = torch.var(z_k, dim=0) + 1e-5
        F_k = 1.0 / var_k
        F_k_smoothed = (F_k + 0.5) ** 0.7
        F_k_norm = F_k_smoothed / torch.sum(F_k_smoothed)
        for c in range(C_tasks[k]):
            df_diag[k, c] = F_k_norm
    acc_fiosr_diag = evaluate_method_online("FIOSR-Diag", C_prototypes, df_diag, test_f, test_l, test_t, batch_size=256, Q_tasks=Q_tasks, seed=seed)
    fiosr_diag_results.append(acc_fiosr_diag)
    
    # 3. Oracle Aligned Diagonal Fisher
    df_oracle = torch.zeros(K, C, d).to(device)
    for k in range(K):
        mask = (cal_t == k)
        z_k = cal_f[mask].view(-1, K, d)[:, k, :]
        z_k_aligned = torch.mm(z_k, Q_tasks[k])
        var_k = torch.var(z_k_aligned, dim=0) + 1e-5
        F_k = 1.0 / var_k
        F_k_smoothed = (F_k + 0.5) ** 0.7
        F_k_norm = F_k_smoothed / torch.sum(F_k_smoothed)
        for c in range(C_tasks[k]):
            df_oracle[k, c] = F_k_norm
    acc_fiosr_rot = evaluate_method_online("FIOSR-Rotated", C_prototypes, df_oracle, test_f, test_l, test_t, batch_size=256, Q_tasks=Q_tasks, seed=seed)
    fiosr_rot_results.append(acc_fiosr_rot)
    
    # 4. Online Estimated Covariance Shrinkage & Alignment (FIOSR-Online)
    U_est_list = []
    df_online = torch.zeros(K, C, d).to(device)
    for k in range(K):
        mask = (cal_t == k)
        z_k = cal_f[mask].view(-1, K, d)[:, k, :]
        labels_k = cal_l[mask]
        
        # Estimate covariance with shrinkage alpha=0.2
        Sigma_k = estimate_covariance(z_k, labels_k, C_tasks[k], alpha=0.2)
        evals, U_k = torch.linalg.eigh(Sigma_k)
        U_est_list.append(U_k)
        
        # Project
        z_k_projected = torch.mm(z_k, U_k)
        
        sq_deviations = torch.zeros(d).to(device)
        df = 0
        for c in range(C_tasks[k]):
            class_mask = (labels_k == c)
            class_samples = z_k_projected[class_mask]
            Nc = len(class_samples)
            if Nc > 1:
                class_mean = class_samples.mean(dim=0)
                sq_deviations += torch.sum((class_samples - class_mean) ** 2, dim=0)
                df += (Nc - 1)
                
        if df > 0:
            var_k = sq_deviations / df + 1e-5
        else:
            var_k = evals + 1e-5
            
        F_k = 1.0 / var_k
        F_k_smoothed = (F_k + 0.5) ** 0.7
        F_k_norm = F_k_smoothed / torch.sum(F_k_smoothed)
        for c in range(C_tasks[k]):
            df_online[k, c] = F_k_norm
            
    acc_fiosr_online = evaluate_method_online("FIOSR-Online", C_prototypes, df_online, test_f, test_l, test_t, batch_size=256, U_est=U_est_list, Q_tasks=Q_tasks, seed=seed)
    fiosr_online_results.append(acc_fiosr_online)
    
    print(f"Seed {seed:<2} | PFSR+MBH: {acc_pfsr:.2f}% | FIOSR-Diag: {acc_fiosr_diag:.2f}% | FIOSR-Oracle: {acc_fiosr_rot:.2f}% | FIOSR-Online (Ours): {acc_fiosr_online:.2f}%")

print("\n=== FINAL ONLINE COVARIANCE ALIGNMENT RESULTS ===")
print(f"PFSR + MBH (Flat Cosine):               {np.mean(pfsr_results):.4f}% +/- {np.std(pfsr_results):.4f}%")
print(f"FIOSR-Diag (Diagonal Fisher):           {np.mean(fiosr_diag_results):.4f}% +/- {np.std(fiosr_diag_results):.4f}%")
print(f"FIOSR-Oracle (Oracle Alignment):        {np.mean(fiosr_rot_results):.4f}% +/- {np.std(fiosr_rot_results):.4f}%")
print(f"FIOSR-Online (Estimated Cov EVD - Ours):{np.mean(fiosr_online_results):.4f}% +/- {np.std(fiosr_online_results):.4f}%")
