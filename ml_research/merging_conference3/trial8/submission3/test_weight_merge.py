import torch
import torch.nn as nn
import numpy as np
import os

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Import elements from run_experiments
import run_experiments
from run_experiments import model, test_x, test_y, test_task, K, num_classes, B_size, qmerge_coeffs_frozen, scale_alignment_factors, get_quantized_zca_coefficients

# Modify forward of SandboxViT to support true weight merging
def merged_forward(self, x, task_idx, active_expert_idx=None, alpha=None, scale_alignment=None, fake_quant_base_bit=None, use_weight_merge=False):
    h = x
    for l_idx, block in enumerate(self.blocks, 1):
        if fake_quant_base_bit is not None:
            if use_weight_merge and block.has_adapters:
                # True parameter-space weight merging
                W_merged = block.W_base.clone()
                for k in range(K):
                    if alpha is not None:
                        coeff = alpha[0, k].item()
                    else:
                        coeff = 1.0 / K
                    adapter = block.adapters[k]
                    W_merged = W_merged + coeff * (adapter.A @ adapter.B)
                
                # Quantize the merged weight
                if fake_quant_base_bit == 4:
                    max_val = torch.max(torch.abs(W_merged), dim=1, keepdim=True)[0]
                    S = max_val / 7.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W_merged / S, -7, 7))
                    W_merged_dequant = Q * S
                elif fake_quant_base_bit == 8:
                    max_val = torch.max(torch.abs(W_merged), dim=1, keepdim=True)[0]
                    S = max_val / 127.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W_merged / S, -127, 127))
                    W_merged_dequant = Q * S
                
                h = h @ W_merged_dequant
            else:
                # Decoupled activation blending path (either specific expert, SA-QAB or uniform fallback)
                W = block.W_base
                if fake_quant_base_bit == 4:
                    max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                    S = max_val / 7.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W / S, -7, 7))
                    W_dequant = Q * S
                elif fake_quant_base_bit == 8:
                    max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                    S = max_val / 127.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W / S, -127, 127))
                    W_dequant = Q * S
                
                base_out = h @ W_dequant
                
                if not block.has_adapters:
                    h = base_out
                    continue
                
                if active_expert_idx is not None:
                    adapter = block.adapters[active_expert_idx]
                    A = adapter.A
                    B = adapter.B
                    
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
                    
                    h = base_out + h @ A_dequant @ B_dequant
                else:
                    blend_out = torch.zeros_like(base_out)
                    for k in range(K):
                        if alpha is not None:
                            coeff = alpha[:, k].unsqueeze(1)
                        else:
                            coeff = 1.0 / K
                        adapter = block.adapters[k]
                        A = adapter.A
                        B = adapter.B
                        
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
                        
                        adapter_out = h @ A_dequant @ B_dequant
                        if scale_alignment is not None:
                            adapter_out = adapter_out * scale_alignment[l_idx][k]
                        blend_out += coeff * adapter_out
                    h = base_out + blend_out
        else:
            h = block(h, active_expert_idx=active_expert_idx, alpha=alpha, scale_factors=scale_alignment[l_idx] if scale_alignment else None)
            
    if isinstance(task_idx, torch.Tensor) and task_idx.ndim > 0:
        logits = torch.zeros(x.shape[0], num_classes)
        for k in range(K):
            mask = (task_idx == k)
            if mask.any():
                logits[mask] = self.heads[k](h[mask])
        return logits
    else:
        return self.heads[task_idx](h)

# Bind the modified forward
import types
model.forward = types.MethodType(merged_forward, model)

# Run homogeneous evaluation of true PMQ 4-bit
print("Running True PMQ 4-bit Homogeneous...")
accuracies = []
model.eval()
with torch.no_grad():
    for k in range(K):
        mask = (test_task == k)
        task_test_x = test_x[mask]
        task_test_y = test_y[mask]
        
        num_batches = int(np.ceil(task_test_x.shape[0] / B_size))
        correct = 0
        total = 0
        
        for b in range(num_batches):
            start = b * B_size
            end = min((b + 1) * B_size, task_test_x.shape[0])
            bx = task_test_x[start:end]
            by = task_test_y[start:end]
            
            # evaluate PMQ with use_weight_merge=True
            logits = model(bx, task_idx=k, fake_quant_base_bit=4, use_weight_merge=True)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.shape[0]
            
        accuracies.append(correct / total * 100.0)

print(f"True PMQ 4-bit Homogeneous Accuracies per task: {accuracies}")
print(f"True PMQ 4-bit Homogeneous Mean: {np.mean(accuracies):.2f}%")

# Run heterogeneous evaluation of true PMQ 4-bit
print("Running True PMQ 4-bit Heterogeneous...")
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
        
        logits = model(bx, task_idx=btask, fake_quant_base_bit=4, use_weight_merge=True)
        preds = logits.argmax(dim=1)
        correct += (preds == by).sum().item()
        total += bx.shape[0]

true_pmq_hetero_acc = correct / total * 100.0
print(f"True PMQ 4-bit Heterogeneous Acc: {true_pmq_hetero_acc:.2f}%")
