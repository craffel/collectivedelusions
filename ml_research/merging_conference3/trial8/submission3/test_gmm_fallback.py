import torch
import torch.nn as nn
import numpy as np
import os

# Import the necessary modules and variables from run_experiments
from run_experiments import (
    model, test_x, test_y, test_task, K, num_classes, B_size, device,
    scale_alignment_factors, gmm_means, gmm_vars, compute_gmm_log_likelihood,
    get_quantized_zca_coefficients
)

print("\n========================================================")
print("RUNNING GMM REJECTION FALLBACK POLICY SIMULATION")
print("========================================================\n")

# Optimal threshold from the experiments
eta = -255.0

def evaluate_sa_qab_with_gmm(fallback_policy="standard", custom_eta=None):
    """
    Evaluates SA-QAB joint heterogeneous accuracy under GMM-based OOD rejection.
    If a sample's log-likelihood is below the threshold eta (false rejection since all samples are ID),
    it is handled based on fallback_policy:
      - "none": No rejection is applied (oracle SA-QAB).
      - "standard": Bypasses adapters completely (alpha set to all zeros), executing on base model only.
      - "soft": Falls back to uniform ensembling (alpha set to 1/K).
    """
    threshold = eta if custom_eta is None else custom_eta
    model.eval()
    correct = 0
    total = 0
    false_rejections = 0
    
    with torch.no_grad():
        num_batches = int(np.ceil(test_x.shape[0] / B_size))
        for b in range(num_batches):
            start = b * B_size
            end = min((b + 1) * B_size, test_x.shape[0])
            bx = test_x[start:end]
            by = test_y[start:end]
            btask = test_task[start:end]
            
            # 1. Forward up to Layer 3 under 4-bit base model to extract features for GMM and ZCA
            h = bx
            for block in model.blocks[:3]:
                W = block.W_base
                max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                S = max_val / 7.0
                S = torch.clamp(S, min=1e-8)
                Q = torch.round(torch.clamp(W / S, -7, 7))
                W_dequant = Q * S
                h = h @ W_dequant
                
            # 2. Compute GMM log-likelihood for each sample in the batch
            log_lh = compute_gmm_log_likelihood(h)
            
            # 3. Compute routing coefficients (standard SA-QAB)
            alpha_orig = get_quantized_zca_coefficients(h, tau=0.001)
            
            # 4. Apply GMM Rejection Gate & Fallback policies sample-by-sample
            alpha_final = alpha_orig.clone()
            
            for i in range(bx.shape[0]):
                if log_lh[i] < threshold:
                    false_rejections += 1
                    if fallback_policy == "standard":
                        # Bypass adapters completely: set routing coefficients to 0
                        alpha_final[i] = 0.0
                    elif fallback_policy == "soft":
                        # Soft fallback: route to uniform ensembling (1/K)
                        alpha_final[i] = 1.0 / K
                    elif fallback_policy == "none":
                        # Do not modify
                        pass
            
            # 5. Forward through remaining layers of SA-QAB (using final alpha)
            logits = model(bx, task_idx=btask, alpha=alpha_final, scale_alignment=scale_alignment_factors, fake_quant_base_bit=4)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.shape[0]
            
    acc = correct / total * 100.0
    frr = false_rejections / total * 100.0
    return acc, frr

# Sweep over thresholds used in the main experiments
thresholds = [-275.0, -265.0, -255.0, -245.0, -235.0]

print("Evaluating SA-QAB with NO GMM Rejection (Upper Bound)...")
acc_none, _ = evaluate_sa_qab_with_gmm(fallback_policy="none")
print(f"  Accuracy: {acc_none:.2f}%\n")

print("| Threshold η | FRR (%) | Overall Std Fallback | Overall Soft Fallback | Rejected SA-QAB Acc (%) | Rejected Std Acc (%) | Rejected Soft Acc (%) |")
print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

def evaluate_detailed_rejected(thresh):
    model.eval()
    rejected_count = 0
    rejected_correct_orig = 0
    rejected_correct_std = 0
    rejected_correct_soft = 0
    
    with torch.no_grad():
        num_batches = int(np.ceil(test_x.shape[0] / B_size))
        for b in range(num_batches):
            start = b * B_size
            end = min((b + 1) * B_size, test_x.shape[0])
            bx = test_x[start:end]
            by = test_y[start:end]
            btask = test_task[start:end]
            
            # Forward up to Layer 3
            h = bx
            for block in model.blocks[:3]:
                W = block.W_base
                max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                S = max_val / 7.0
                S = torch.clamp(S, min=1e-8)
                Q = torch.round(torch.clamp(W / S, -7, 7))
                W_dequant = Q * S
                h = h @ W_dequant
                
            log_lh = compute_gmm_log_likelihood(h)
            alpha_orig = get_quantized_zca_coefficients(h, tau=0.001)
            
            # Run evaluations
            # 1. Original SA-QAB
            logits_orig = model(bx, task_idx=btask, alpha=alpha_orig, scale_alignment=scale_alignment_factors, fake_quant_base_bit=4)
            preds_orig = logits_orig.argmax(dim=1)
            
            # 2. Standard Fallback (alpha = 0 for rejected)
            alpha_std = alpha_orig.clone()
            # 3. Soft Fallback (alpha = 1/K for rejected)
            alpha_soft = alpha_orig.clone()
            
            rejected_mask = (log_lh < thresh)
            if rejected_mask.any():
                alpha_std[rejected_mask] = 0.0
                alpha_soft[rejected_mask] = 1.0 / K
                
            logits_std = model(bx, task_idx=btask, alpha=alpha_std, scale_alignment=scale_alignment_factors, fake_quant_base_bit=4)
            preds_std = logits_std.argmax(dim=1)
            
            logits_soft = model(bx, task_idx=btask, alpha=alpha_soft, scale_alignment=scale_alignment_factors, fake_quant_base_bit=4)
            preds_soft = logits_soft.argmax(dim=1)
            
            for i in range(bx.shape[0]):
                if log_lh[i] < thresh:
                    rejected_count += 1
                    if preds_orig[i] == by[i]:
                        rejected_correct_orig += 1
                    if preds_std[i] == by[i]:
                        rejected_correct_std += 1
                    if preds_soft[i] == by[i]:
                        rejected_correct_soft += 1
                        
    acc_orig = (rejected_correct_orig / rejected_count * 100.0) if rejected_count > 0 else 0.0
    acc_std = (rejected_correct_std / rejected_count * 100.0) if rejected_count > 0 else 0.0
    acc_soft = (rejected_correct_soft / rejected_count * 100.0) if rejected_count > 0 else 0.0
    return rejected_count, acc_orig, acc_std, acc_soft

for thresh in thresholds:
    acc_std_overall, frr_std = evaluate_sa_qab_with_gmm(fallback_policy="standard", custom_eta=thresh)
    acc_soft_overall, _ = evaluate_sa_qab_with_gmm(fallback_policy="soft", custom_eta=thresh)
    
    rejected_count, acc_orig_rej, acc_std_rej, acc_soft_rej = evaluate_detailed_rejected(thresh)
    print(f"| {thresh:5.1f} | {frr_std:5.1f}% | {acc_std_overall:5.2f}% | {acc_soft_overall:5.2f}% | {acc_orig_rej:5.1f}% | {acc_std_rej:5.1f}% | {acc_soft_rej:5.1f}% |")

print("\n========================================================\n")
