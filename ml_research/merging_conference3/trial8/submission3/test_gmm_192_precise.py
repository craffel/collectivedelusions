import torch
import numpy as np
from run_experiments import model, test_x, test_task, calib_x, calib_task, test_is_ood

model.eval()

def get_layer3_feats(x_batch):
    with torch.no_grad():
        h = x_batch
        for block in model.blocks[:3]:
            W = block.W_base
            max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
            S = max_val / 7.0
            S = torch.clamp(S, min=1e-8)
            Q = torch.round(torch.clamp(W / S, -7, 7))
            W_dequant = Q * S
            h = h @ W_dequant
            
        # Quantize h
        max_h = torch.max(torch.abs(h), dim=-1, keepdim=True)[0]
        S_h = max_h / 127.0
        S_h = torch.clamp(S_h, min=1e-8)
        Q_h = torch.round(torch.clamp(h / S_h, -127, 127))
        h_q = Q_h * S_h
        return h_q

# Extract features for training (calibration set)
calib_feats = get_layer3_feats(calib_x)
test_feats = get_layer3_feats(test_x)

# Fit GMM (192D diagonal)
K = 4
gmm_means = {}
gmm_vars = {}
for k in range(K):
    mask = (calib_task == k)
    feats_k = calib_feats[mask]
    gmm_means[k] = feats_k.mean(dim=0)
    gmm_vars[k] = feats_k.var(dim=0) + 1e-4 # add ridge term

def compute_gmm_log_likelihood_192(h_q):
    log_probs = []
    for k in range(K):
        m = gmm_means[k]
        v = gmm_vars[k]
        diff = h_q - m
        # 192D diagonal log-likelihood
        log_density = -0.5 * torch.sum(torch.log(2 * np.pi * v) + (diff ** 2) / v, dim=-1)
        log_probs.append(log_density)
    log_probs = torch.stack(log_probs, dim=1)
    return torch.logsumexp(log_probs, dim=1) - np.log(K)

# Compute log-likelihoods for ID test set
id_lh = compute_gmm_log_likelihood_192(test_feats)

# Compute log-likelihoods for OOD test set
ood_x = torch.randn(250, 192)
ood_feats = get_layer3_feats(ood_x)
ood_lh = compute_gmm_log_likelihood_192(ood_feats)

all_lh = torch.cat([id_lh, ood_lh])
is_ood = torch.cat([torch.zeros_like(id_lh, dtype=torch.bool), torch.ones_like(ood_lh, dtype=torch.bool)])

thresholds = np.linspace(-340.0, -300.0, 41)
print("\nPrecise 192D GMM Threshold Sweep:")
for eta in thresholds:
    rejected = (all_lh < eta)
    tpr = (rejected & is_ood).sum().item() / is_ood.sum().item() * 100.0
    fpr = (rejected & (~is_ood)).sum().item() / (~is_ood).sum().item() * 100.0
    print(f"  Threshold η: {eta:7.1f} | TPR: {tpr:5.1f}% | FPR: {fpr:5.1f}%")
