import torch
import torch.nn as nn
import numpy as np
import copy
from run_experiments import (
    model, test_x, test_y, test_task, calib_x, calib_task, K, num_classes, B_size, device,
    scale_alignment_factors, fp16_state_dict, get_quantized_zca_coefficients
)

print("\n========================================================")
print("EVALUATING TRAINING-FREE PTQ ALTERNATIVES FOR SA-QAB")
print("========================================================\n")

# Make sure we load the unquantized model first
model.load_state_dict(fp16_state_dict)

def evaluate_sa_qab_custom(scale_alignment=None, bias_corrections=None, smoothquant_scales=None):
    """
    Evaluates SA-QAB joint heterogeneous accuracy with custom scale alignment, bias correction, or channel pre-scaling.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        num_batches = int(np.ceil(test_x.shape[0] / B_size))
        for b in range(num_batches):
            start = b * B_size
            end = min((b + 1) * B_size, test_x.shape[0])
            bx = test_x[start:end]
            by = test_y[start:end]
            btask = test_task[start:end]
            
            # 1. Forward up to Layer 3 under 4-bit base model
            h = bx
            for l_idx, block in enumerate(model.blocks[:3], 1):
                W = block.W_base
                
                # Apply SmoothQuant scaling to W if present
                if smoothquant_scales is not None and l_idx in smoothquant_scales:
                    # h = h / s, W_scaled = s * W
                    s = smoothquant_scales[l_idx]
                    h_scaled = h / s
                    W_scaled = s.unsqueeze(1) * W
                else:
                    h_scaled = h
                    W_scaled = W
                
                max_val = torch.max(torch.abs(W_scaled), dim=1, keepdim=True)[0]
                S = max_val / 7.0
                S = torch.clamp(S, min=1e-8)
                Q = torch.round(torch.clamp(W_scaled / S, -7, 7))
                W_dequant = Q * S
                
                h = h_scaled @ W_dequant
                
                if bias_corrections is not None and l_idx in bias_corrections:
                    h = h + bias_corrections[l_idx]
            
            # Get quantized ZCA routing coefficients
            alpha = get_quantized_zca_coefficients(h, tau=0.001)
            
            # Forward through late layers 4 to 12
            for l_idx, block in enumerate(model.blocks[3:], 4):
                W = block.W_base
                
                # Apply SmoothQuant scaling to W if present
                if smoothquant_scales is not None and l_idx in smoothquant_scales:
                    s = smoothquant_scales[l_idx]
                    h_scaled = h / s
                    W_scaled = s.unsqueeze(1) * W
                else:
                    h_scaled = h
                    W_scaled = W
                    
                max_val = torch.max(torch.abs(W_scaled), dim=1, keepdim=True)[0]
                S = max_val / 7.0
                S = torch.clamp(S, min=1e-8)
                Q = torch.round(torch.clamp(W_scaled / S, -7, 7))
                W_dequant = Q * S
                
                base_out = h_scaled @ W_dequant
                
                # Blended expert path with INT8 adapter quantization
                blend_out = torch.zeros_like(base_out)
                for k in range(K):
                    coeff = alpha[:, k].unsqueeze(1)
                    adapter = block.adapters[k]
                    A = adapter.A
                    B = adapter.B
                    
                    # Quantize A & B to INT8
                    max_A = torch.max(torch.abs(A))
                    S_A = max_A / 127.0
                    S_A = torch.clamp(S_A, min=1e-8)
                    Q_A = torch.round(torch.clamp(A / S_A, -127, 127))
                    A_dequant = Q_A * S_A
                    
                    max_B = torch.max(torch.abs(B))
                    S_B = max_B / 127.0
                    S_B = torch.clamp(S_B, min=1e-8)
                    Q_B = torch.round(torch.clamp(B / S_B, -127, 127))
                    B_dequant = Q_B * S_B
                    
                    adapter_out = h_scaled @ A_dequant @ B_dequant
                    if scale_alignment is not None:
                        adapter_out = adapter_out * scale_alignment[l_idx][k]
                        
                    blend_out += coeff * adapter_out
                
                h = torch.nn.functional.gelu(base_out + blend_out)
                
                if bias_corrections is not None and l_idx in bias_corrections:
                    h = h + bias_corrections[l_idx]
            
            # Predict
            logits = torch.zeros(bx.shape[0], num_classes)
            for k in range(K):
                mask = (btask == k)
                if mask.any():
                    logits[mask] = model.heads[k](h[mask])
                    
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.shape[0]
            
    return correct / total * 100.0

# -------------------------------------------------------------------------
# 1. EVALUATING BASELINE PTQ SCENARIOS
# -------------------------------------------------------------------------
model.load_state_dict(fp16_state_dict)

print("Running Direct PTQ SA-QAB (Unquantized Model, NO Scale Alignment, NO QAT)...")
acc_direct = evaluate_sa_qab_custom(scale_alignment=None)
print(f"  Accuracy: {acc_direct:.2f}%\n")

print("Running Standard QSR PTQ SA-QAB (Unquantized Model, WITH Scale Alignment, NO QAT)...")
acc_qsr = evaluate_sa_qab_custom(scale_alignment=scale_alignment_factors)
print(f"  Accuracy: {acc_qsr:.2f}%\n")


# -------------------------------------------------------------------------
# 2. COMPUTING ACTIVATION BIAS CORRECTION (ABC)
# -------------------------------------------------------------------------
print("Computing Activation Bias Correction (ABC) over calibration set...")
bias_corrections = {}
model.load_state_dict(fp16_state_dict)

with torch.no_grad():
    # We pass the entire calibration set to compute expected representations
    # A. Full precision reference pass
    h_fp = calib_x
    calib_h_fp_layers = {}
    for l_idx, block in enumerate(model.blocks, 1):
        if l_idx < 4:
            h_fp = block(h_fp)
        else:
            base_out = h_fp @ block.W_base
            blend_out = torch.zeros_like(base_out)
            # Replicate activation blending for calibration
            for k in range(K):
                # Calculate oracle routing from task index
                mask = (calib_task == k)
                coeff = torch.zeros(calib_x.shape[0], 1)
                coeff[mask] = 1.0
                adapter_out = block.adapters[k](h_fp)
                blend_out += coeff * adapter_out
            h_fp = torch.nn.functional.gelu(base_out + blend_out)
        calib_h_fp_layers[l_idx] = h_fp.clone()

    # B. Quantized pass to compute mean shift
    h_q = calib_x
    for l_idx, block in enumerate(model.blocks, 1):
        W = block.W_base
        max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
        S = max_val / 7.0
        S = torch.clamp(S, min=1e-8)
        Q = torch.round(torch.clamp(W / S, -7, 7))
        W_dequant = Q * S
        
        if l_idx < 4:
            h_q = h_q @ W_dequant
            h_q = torch.nn.functional.gelu(h_q)
        else:
            base_out = h_q @ W_dequant
            blend_out = torch.zeros_like(base_out)
            for k in range(K):
                mask = (calib_task == k)
                coeff = torch.zeros(calib_x.shape[0], 1)
                coeff[mask] = 1.0
                adapter = block.adapters[k]
                A = adapter.A
                B = adapter.B
                
                # Quantize A & B to INT8
                max_A = torch.max(torch.abs(A))
                S_A = max_A / 127.0
                S_A = torch.clamp(S_A, min=1e-8)
                Q_A = torch.round(torch.clamp(A / S_A, -127, 127))
                A_dequant = Q_A * S_A
                
                max_B = torch.max(torch.abs(B))
                S_B = max_B / 127.0
                S_B = torch.clamp(S_B, min=1e-8)
                Q_B = torch.round(torch.clamp(B / S_B, -127, 127))
                B_dequant = Q_B * S_B
                
                adapter_out = h_q @ A_dequant @ B_dequant
                adapter_out = adapter_out * scale_alignment_factors[l_idx][k]
                blend_out += coeff * adapter_out
            h_q = torch.nn.functional.gelu(base_out + blend_out)
        
        # Bias vector is the mean difference of activations
        bias = calib_h_fp_layers[l_idx].mean(dim=0) - h_q.mean(dim=0)
        bias_corrections[l_idx] = bias
        # Apply bias correction during calibration tracking
        h_q = h_q + bias

print("Activation Bias Correction (ABC) computed successfully.")
print("Evaluating SA-QAB with ABC (NO QAT)...")
acc_abc = evaluate_sa_qab_custom(scale_alignment=scale_alignment_factors, bias_corrections=bias_corrections)
print(f"  Accuracy with ABC: {acc_abc:.2f}%\n")


# -------------------------------------------------------------------------
# 3. COMPUTING ACTIVATION-AWARE PRE-SCALING (SMOOTHQUANT-LIKE)
# -------------------------------------------------------------------------
print("Computing SmoothQuant-like Activation-Aware Pre-Scaling over calibration set...")
smoothquant_scales = {}
alpha_sq = 0.5 # SmoothQuant migration hyperparameter

model.load_state_dict(fp16_state_dict)
with torch.no_grad():
    # Pass calibration set through full precision model to find activation maximums per channel
    h_fp = calib_x
    for l_idx, block in enumerate(model.blocks, 1):
        # 1. Compute per-channel max activation
        # h_fp shape: (B, D)
        max_act = torch.max(torch.abs(h_fp), dim=0)[0] # (D,)
        max_act = torch.clamp(max_act, min=1e-5)
        
        # 2. Compute per-channel (row-wise) max weight
        W = block.W_base # (D, D)
        max_wt = torch.max(torch.abs(W), dim=1)[0] # (D,)
        max_wt = torch.clamp(max_wt, min=1e-5)
        
        # 3. Compute SmoothQuant scaling factors
        # s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
        s = torch.pow(max_act, alpha_sq) / torch.pow(max_wt, 1.0 - alpha_sq)
        s = torch.clamp(s, min=1e-3)
        smoothquant_scales[l_idx] = s
        
        # Forward pass for next layer calibration
        if l_idx < 4:
            h_fp = block(h_fp)
        else:
            base_out = h_fp @ block.W_base
            blend_out = torch.zeros_like(base_out)
            for k in range(K):
                mask = (calib_task == k)
                coeff = torch.zeros(calib_x.shape[0], 1)
                coeff[mask] = 1.0
                adapter_out = block.adapters[k](h_fp)
                blend_out += coeff * adapter_out
            h_fp = torch.nn.functional.gelu(base_out + blend_out)

print("SmoothQuant-like pre-scaling computed successfully.")
print("Evaluating SA-QAB with Activation Pre-scaling (NO QAT)...")
acc_sq = evaluate_sa_qab_custom(scale_alignment=scale_alignment_factors, smoothquant_scales=smoothquant_scales)
print(f"  Accuracy with SmoothQuant: {acc_sq:.2f}%\n")


# -------------------------------------------------------------------------
# 4. HYBRID PTQ: BIAS CORRECTION + SMOOTHQUANT (NO QAT)
# -------------------------------------------------------------------------
print("Evaluating SA-QAB with BOTH SmoothQuant Pre-scaling and Bias Correction (ABC, NO QAT)...")
acc_hybrid = evaluate_sa_qab_custom(scale_alignment=scale_alignment_factors, bias_corrections=bias_corrections, smoothquant_scales=smoothquant_scales)
print(f"  Hybrid PTQ Accuracy: {acc_hybrid:.2f}%\n")

# Report summary table
print("------------------------------------------------------------")
print("SUMMARY OF TRAINING-FREE PTQ UPGRADE RESULTS")
print("------------------------------------------------------------")
print(f"  1. Direct PTQ SA-QAB (No Alignment)       : {acc_direct:.2f}%")
print(f"  2. Standard QSR PTQ SA-QAB (Our baseline) : {acc_qsr:.2f}%")
print(f"  3. QSR + Activation Bias Correction (ABC) : {acc_abc:.2f}%")
print(f"  4. QSR + SmoothQuant Pre-scaling          : {acc_sq:.2f}%")
print(f"  5. QSR + ABC + SmoothQuant Hybrid         : {acc_hybrid:.2f}%")
print("------------------------------------------------------------")
