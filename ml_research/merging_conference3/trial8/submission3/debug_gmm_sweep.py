import torch
import numpy as np
from run_experiments import test_log_likelihoods, test_is_ood

# Let's sweep thresholds from 0.0 to 4.0
thresholds = np.linspace(0.0, 4.0, 17)
print("GMM Threshold Sweep:")
for eta in thresholds:
    rejected = (test_log_likelihoods < eta)
    tp = (rejected & test_is_ood).sum().item()
    tpr = tp / test_is_ood.sum().item() * 100.0
    fp = (rejected & (~test_is_ood)).sum().item()
    fpr = fp / (~test_is_ood).sum().item() * 100.0
    print(f"  Threshold {eta:.2f} | TPR: {tpr:.1f}% | FPR: {fpr:.1f}%")
