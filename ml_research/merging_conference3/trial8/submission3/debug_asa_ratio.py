import torch
import numpy as np
from run_experiments import model, test_x, test_y, test_task, calib_x, calib_task, get_quantized_zca_coefficients, K, num_classes

# Recompute scale alignment factors using the FP / Quant adapter norm ratio
scale_alignment_factors_ratio = {l: [1.0]*K for l in range(1, 13)}

model.eval()
with torch.no_grad():
    for k in range(K):
        mask = (calib_task == k)
        task_cal_x = calib_x[mask]
        
        h_base = task_cal_x
        for l_idx, block in enumerate(model.blocks, 1):
            if l_idx < 4:
                h_base = block(h_base)
            else:
                base_out = h_base @ block.W_base
                
                # FP adapter out
                adapter_fp_out = block.adapters[k](h_base)
                norm_adapter_fp = torch.norm(adapter_fp_out, p=2, dim=-1).mean().item()
                
                # Quantized adapter out
                A = block.adapters[k].A
                B = block.adapters[k].B
                
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
                
                adapter_quant_out = h_base @ A_dequant @ B_dequant
                norm_adapter_quant = torch.norm(adapter_quant_out, p=2, dim=-1).mean().item()
                
                # Scale alignment ratio
                beta = norm_adapter_fp / max(norm_adapter_quant, 1e-8)
                scale_alignment_factors_ratio[l_idx][k] = beta
                
                h_base = base_out + adapter_fp_out

print("--- Computed Quant-to-FP Scale Alignment Factors ---")
for l in range(4, 13):
    print(f"Layer {l}: {[f'{b:.6f}' for b in scale_alignment_factors_ratio[l]]}")

# Evaluate
correct = 0
total = 0
B_size = 256

with torch.no_grad():
    num_batches = int(np.ceil(test_x.shape[0] / B_size))
    for b in range(num_batches):
        start = b * B_size
        end = min((b + 1) * B_size, test_x.shape[0])
        bx = test_x[start:end]
        by = test_y[start:end]
        btask = test_task[start:end]
        
        h = bx
        for block in model.blocks[:3]:
            W = block.W_base
            max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
            S = max_val / 7.0
            S = torch.clamp(S, min=1e-8)
            Q = torch.round(torch.clamp(W / S, -7, 7))
            W_dequant = Q * S
            h = h @ W_dequant
            
        alpha = get_quantized_zca_coefficients(h, tau=0.001)
        
        logits = model(bx, task_idx=btask, alpha=alpha, scale_alignment=scale_alignment_factors_ratio, fake_quant_base_bit=4)
        preds = logits.argmax(dim=1)
        correct += (preds == by).sum().item()
        total += bx.shape[0]

accuracy_ratio = correct / total * 100.0
print(f"SA-QAB Joint Accuracy with FP/Quant ASA ratio: {accuracy_ratio:.2f}%")
