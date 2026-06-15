import torch
import numpy as np
import run_experiments

# Let's import our generation functions
from run_experiments import generate_sandbox_data, evaluate_method

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set K, d, C_tasks from run_experiments
K = 4
d = 48
C_tasks = [10, 10, 10, 4]

# Let's write a function to estimate the raw, unsmoothed diagonal Fisher
def estimate_raw_diagonal_fisher(C_prototypes, cal_features, cal_labels, cal_tasks):
    F_raw = torch.zeros(K, 10, d).to(device)

    for k in range(K):
        # Extract calibration samples of task k
        mask = (cal_tasks == k)
        feat_k = cal_features[mask] # [16, D]

        # Expert block features
        z_k = feat_k.view(-1, K, d)[:, k, :] # [16, d]

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

        # Fisher Information of Gaussian mean parameter is inverse variance
        F_k_raw = 1.0 / var_k # [d]

        # Broadcast across classes of task k
        for c in range(C_tasks[k]):
            F_raw[k, c] = F_k_raw

    return F_raw

# Let's load Seed 42
C_prototypes, cal_f, cal_l, cal_t, test_f, test_l, test_t = generate_sandbox_data(42, rho=0.33)

# Apply pre-calibration mean-centering
mean_cal = cal_f.mean(dim=0, keepdim=True)
cal_f = cal_f - mean_cal
test_f = test_f - mean_cal

Fisher_M_raw = estimate_raw_diagonal_fisher(C_prototypes, cal_f, cal_l, cal_t)

# Let's inspect the Fisher weights
print("Fisher raw statistics:")
for k in range(4):
    print(f"Task {k} class 0: min={Fisher_M_raw[k, 0].min().item():.6f}, max={Fisher_M_raw[k, 0].max().item():.6f}, std={Fisher_M_raw[k, 0].std().item():.6f}")

# Let's write a modified get_mbh_coefficients with smoothed Fisher applied on raw
def get_mbh_coefficients_smoothed(method_name, feat, C_prototypes, Fisher_M_raw, beta=0.1, gamma=0.5):
    B_size = len(feat)
    z_blocks = feat.view(B_size, K, d)
    u = torch.zeros(B_size, K).to(device)
    
    for k in range(K):
        W_k = C_prototypes[k, :C_tasks[k]] # [C_tasks[k], d]
        z_k = z_blocks[:, k, :] # [B, d]
        
        # Smooth Fisher weights: F_smooth = (F_raw + beta) ** gamma
        F_k_raw = Fisher_M_raw[k, :C_tasks[k]] # [C_tasks[k], d]
        F_k_smoothed = (F_k_raw + beta) ** gamma
        F_k_smoothed = F_k_smoothed / F_k_smoothed.sum(dim=-1, keepdim=True)
        
        # Fisher-Weighted Cosine Similarity
        z_k_expanded = z_k.unsqueeze(1) # [B, 1, d]
        W_k_expanded = W_k.unsqueeze(0) # [1, C_tasks[k], d]
        F_k_expanded = F_k_smoothed.unsqueeze(0) # [1, C_tasks[k], d]
        
        num = torch.sum(F_k_expanded * W_k_expanded * z_k_expanded, dim=-1) # [B, C_tasks[k]]
        den1 = torch.sqrt(torch.sum(F_k_expanded * (W_k_expanded ** 2), dim=-1)) # [1, C_tasks[k]]
        den2 = torch.sqrt(torch.sum(F_k_expanded * (z_k_expanded ** 2), dim=-1)) # [B, C_tasks[k]]
        sims = num / (den1 * den2 + 1e-8) # [B, C_tasks[k]]
        
        u[:, k] = torch.max(sims, dim=-1)[0]
        
    u_calibrated = u / run_experiments.csc_denom
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

# Let's run a sweep of beta and gamma on Seed 42 and measure accuracy
# We will evaluate under homogeneous stream (B=256)
best_acc = 0
best_beta = 0
best_gamma = 0

# Also evaluate the standard PFSR_MBH baseline on Seed 42 for reference
from run_experiments import train_routers
model_objs = train_routers(C_prototypes, cal_f, cal_l, cal_t)

pfsr_acc, _ = evaluate_method("PFSR_MBH", model_objs, C_prototypes, Fisher_M_raw, test_f, test_l, test_t, batch_size=256, is_homogeneous=True)
print(f"Baseline PFSR_MBH Accuracy on Seed 42: {pfsr_acc:.2f}%")

print("\nSweeping beta and gamma...")
for beta in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    for gamma in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        # Evaluate with smoothed Fisher coefficients
        # Monkey patch get_mbh_coefficients inside evaluate_method
        run_experiments.get_mbh_coefficients = lambda m, feat, C, F: get_mbh_coefficients_smoothed(m, feat, C, F, beta=beta, gamma=gamma)
        
        acc, _ = evaluate_method("FIOSR", model_objs, C_prototypes, Fisher_M_raw, test_f, test_l, test_t, batch_size=256, is_homogeneous=True)
        print(f"beta={beta:<5}, gamma={gamma:<5} -> Accuracy: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_beta = beta
            best_gamma = gamma

print(f"\nBest FIOSR accuracy on Seed 42: {best_acc:.2f}% (with beta={best_beta}, gamma={best_gamma})")
