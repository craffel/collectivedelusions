import test_improved_sable as t
import torch
import torch.nn.functional as F
import numpy as np

# Set seeds
t.set_seeds(42)

print("Running SABLE Mid-Layer Routing (L_route) Ablation Sweep...")
# We will evaluate SABLE with routing layer L_route in {0, 2, 4, 6, 8, 10, 12}
# When routing at L_route, layers l < L_route run strictly through the base model.
# At layer L_route, we compute coefficients from the current feature representation.
# For layers l >= L_route, we apply SABLE activation blending.

L_route_values = [0, 2, 4, 6, 8, 10, 12]

for L_route in L_route_values:
    with torch.no_grad():
        # Pass through base model until L_route
        feat = t.test_x
        for l in range(L_route):
            W_base = t.base_backbone.layers[l].weight.data
            b_base = t.base_backbone.layers[l].bias.data
            feat = F.relu(F.linear(feat, W_base, b_base))
            
        # At layer L_route, compute routing coefficients from current feat
        coeffs = t.pfsr_coefficients(feat)
        
        # Apply SABLE blending from L_route onwards
        active_expert_mask = (coeffs > 1e-12).any(dim=0)
        coeffs_reshaped = coeffs.t().unsqueeze(-1)
        
        for l in range(L_route, t.L):
            W_base = t.base_backbone.layers[l].weight.data
            b_base = t.base_backbone.layers[l].bias.data
            H_base = F.linear(feat, W_base, b_base)
            
            H_experts = torch.zeros(t.K, feat.shape[0], t.D, device=feat.device)
            for k in range(t.K):
                if active_expert_mask[k]:
                    A_kl = t.A_adapters[k][l]
                    B_kl = t.B_adapters[k][l]
                    delta_b_kl = t.delta_biases[k][l]
                    proj = torch.matmul(feat, B_kl.t())
                    out = torch.matmul(proj, A_kl.t()) + delta_b_kl
                    H_experts[k] = out
            
            H_blended = torch.sum(coeffs_reshaped * H_experts, dim=0)
            feat = F.relu(H_base + H_blended)
            
        # Head Blending
        logits = t.get_blended_head_logits(feat, coeffs)
        acc = t.compute_acc(logits, t.test_y)
        print(f"  L_route = {L_route:<2} (first {L_route} layers unadapted): SABLE Accuracy = {acc:.2f}%")
