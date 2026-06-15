import torch
import numpy as np
from run_experiments import model, test_x, test_task, get_fp_zca_coefficients, get_quantized_zca_coefficients, scale_alignment_factors

model.eval()
with torch.no_grad():
    # 1. Inspect scale alignment factors
    print("--- Scale Alignment Factors beta_k^(l) ---")
    for l in range(4, 13):
        print(f"Layer {l}: {[f'{b:.4f}' for b in scale_alignment_factors[l]]}")
        
    # 2. Inspect adapter parameter norms
    print("\n--- Adapter Weight Norms ---")
    for k in range(4):
        norms_A = []
        norms_B = []
        for l in range(4, 13):
            block = model.blocks[l-1]
            adapter = block.adapters[k]
            norms_A.append(torch.norm(adapter.A).item())
            norms_B.append(torch.norm(adapter.B).item())
        print(f"Expert {k}:")
        print(f"  A norms: {[f'{n:.4f}' for n in norms_A]}")
        print(f"  B norms: {[f'{n:.4f}' for n in norms_B]}")
        
    # 3. Inspect routing coefficients
    print("\n--- Routing Coefficients (First 5 samples of each task) ---")
    for k in range(4):
        mask = (test_task == k)
        bx = test_x[mask][:5]
        
        # FP routing
        h_fp = bx
        for block in model.blocks[:3]:
            h_fp = block(h_fp)
        alpha_fp = get_fp_zca_coefficients(h_fp, tau=0.001)
        
        # Quant routing
        h_q = bx
        for block in model.blocks[:3]:
            W = block.W_base
            max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
            S = max_val / 7.0
            S = torch.clamp(S, min=1e-8)
            Q = torch.round(torch.clamp(W / S, -7, 7))
            W_dequant = Q * S
            h_q = h_q @ W_dequant
        alpha_q = get_quantized_zca_coefficients(h_q, tau=0.001)
        
        print(f"Task {k} - FP Routing coefficients:")
        print(alpha_fp)
        print(f"Task {k} - Quant Routing coefficients:")
        print(alpha_q)
