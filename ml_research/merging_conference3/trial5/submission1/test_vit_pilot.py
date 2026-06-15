import sys
import os
sys.path.insert(0, os.path.abspath("./custom_libs"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTForImageClassification
from torch.func import functional_call
import numpy as np

def test_vit_pilot():
    print("Running test_vit_pilot...")
    model_name = "google/vit-base-patch16-224"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = ViTForImageClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    ).to(device)
    base_model.eval()

    inputs_task1 = torch.randn(8, 3, 224, 224)
    labels_task1 = torch.randint(0, 2, (8,))
    inputs_task2 = torch.randn(8, 3, 224, 224)
    labels_task2 = torch.randint(0, 2, (8,))

    expert1 = ViTForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True).to(device)
    expert2 = ViTForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True).to(device)

    with torch.no_grad():
        for p1, p2 in zip(expert1.parameters(), expert2.parameters()):
            p1.add_(torch.randn_like(p1) * 0.005)
            p2.add_(torch.randn_like(p2) * 0.005)

    expert1.eval()
    expert2.eval()

    L = 12
    v1_layers = []
    v2_layers = []
    base_layers = []

    for l in range(L):
        layer_base = getattr(base_model.vit.encoder.layer, str(l))
        layer_exp1 = getattr(expert1.vit.encoder.layer, str(l))
        layer_exp2 = getattr(expert2.vit.encoder.layer, str(l))
        
        v1_p = {n: p_exp1 - p_base for (n, p_base), p_exp1 in zip(layer_base.named_parameters(), layer_exp1.parameters())}
        v2_p = {n: p_exp2 - p_base for (n, p_base), p_exp2 in zip(layer_base.named_parameters(), layer_exp2.parameters())}
        base_p = {n: p_base for n, p_base in layer_base.named_parameters()}
        
        v1_layers.append(v1_p)
        v2_layers.append(v2_p)
        base_layers.append(base_p)

    c = torch.ones(L, device=device)

    def get_merged_params(coeffs, target_expert):
        merged_params = {}
        for name, param in target_expert.named_parameters():
            is_merged = False
            for l in range(L):
                prefix = f"vit.encoder.layer.{l}."
                if name.startswith(prefix):
                    is_merged = True
                    p_name = name[len(prefix):]
                    merged_p = base_layers[l][p_name] + coeffs[0, l]*v1_layers[l][p_name] + coeffs[1, l]*v2_layers[l][p_name]
                    merged_params[name] = merged_p
                    break
            if not is_merged:
                merged_params[name] = param
        return merged_params

    # Run adaptation
    coeffs_rcr = (torch.ones((2, L), device=device) * 0.5).requires_grad_(True)
    optimizer = torch.optim.Adam([coeffs_rcr], lr=0.5)

    # 1 optimization step to verify gradients and backward pass
    optimizer.zero_grad()
    m_p1 = get_merged_params(coeffs_rcr, expert1)
    outputs = functional_call(expert1, m_p1, kwargs={"pixel_values": inputs_task1})
    probs = torch.softmax(outputs.logits, dim=-1)
    loss_tta = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
    
    loss_reg = 0.0
    for l in range(1, L):
        loss_reg += torch.sqrt(c[l] * c[l-1]) * ((coeffs_rcr[:, l] - coeffs_rcr[:, l-1])**2).sum()
    
    loss_joint = loss_tta + 2.0 * loss_reg
    loss_joint.backward()
    optimizer.step()

    print("test_vit_pilot completed successfully!")
    assert coeffs_rcr.grad is not None, "Gradients should be populated."

if __name__ == "__main__":
    test_vit_pilot()
