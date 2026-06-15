import sys
import os
sys.path.insert(0, os.path.abspath("./custom_libs"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTForImageClassification
from torch.func import functional_call
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def main():
    print("="*60)
    print("REAL-WORLD ViT-B/16 MODEL MERGING PILOT STUDY (L=12)")
    print("="*60)

    model_name = "google/vit-base-patch16-224"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load pre-trained ViT-B/16 model
    print("Loading pre-trained ViT-B/16 model...")
    base_model = ViTForImageClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    ).to(device)
    base_model.eval()

    # 2. Generate synthetic image data for 2 distinct classification tasks
    print("Generating synthetic datasets...")
    inputs_task1 = torch.randn(20, 3, 224, 224)
    labels_task1 = torch.randint(0, 2, (20,))

    inputs_task2 = torch.randn(20, 3, 224, 224)
    labels_task2 = torch.randint(0, 2, (20,))

    ds1 = TensorDataset(inputs_task1, labels_task1)
    ds2 = TensorDataset(inputs_task2, labels_task2)

    dl1 = DataLoader(ds1, batch_size=4, shuffle=False)
    dl2 = DataLoader(ds2, batch_size=4, shuffle=False)

    # 3. Simulate two expert models by perturbing the base model (instant on CPU)
    print("Simulating Expert 1 and Expert 2 by introducing specialized perturbations...")
    expert1 = ViTForImageClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    ).to(device)
    
    expert2 = ViTForImageClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    ).to(device)

    with torch.no_grad():
        # Perturb parameters slightly to represent localized task specialization
        for p1, p2 in zip(expert1.parameters(), expert2.parameters()):
            # Small perturbations to simulate fine-tuned state
            p1.add_(torch.randn_like(p1) * 0.005)
            p2.add_(torch.randn_like(p2) * 0.005)

    expert1.eval()
    expert2.eval()

    # Define task vectors for the 12 encoder layer blocks (L=12)
    L = 12
    K = 2

    v1_layers = []
    v2_layers = []
    base_layers = []

    for l in range(L):
        layer_base = getattr(base_model.vit.encoder.layer, str(l))
        layer_exp1 = getattr(expert1.vit.encoder.layer, str(l))
        layer_exp2 = getattr(expert2.vit.encoder.layer, str(l))
        
        # Collect parameters
        v1_p = {n: p_exp1 - p_base for (n, p_base), p_exp1 in zip(layer_base.named_parameters(), layer_exp1.parameters())}
        v2_p = {n: p_exp2 - p_base for (n, p_base), p_exp2 in zip(layer_base.named_parameters(), layer_exp2.parameters())}
        base_p = {n: p_base for n, p_base in layer_base.named_parameters()}
        
        v1_layers.append(v1_p)
        v2_layers.append(v2_p)
        base_layers.append(base_p)

    # 4. Estimate base FIM trace (base curvature) for the 12 layers of ViT
    print("Estimating pre-trained base model curvature (FIM diagonal trace)...")
    c_raw = torch.zeros(L, device=device)
    base_model.zero_grad()
    
    # Use calibration inputs from both tasks
    cal_inputs = torch.cat([inputs_task1[:4], inputs_task2[:4]], dim=0).to(device)
    outputs = base_model(pixel_values=cal_inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    
    for i in range(len(cal_inputs)):
        y_sample = torch.multinomial(probs[i], 1).item()
        loss = -torch.log(probs[i, y_sample] + 1e-8)
        grads = torch.autograd.grad(loss, base_model.parameters(), retain_graph=True, allow_unused=True)
        
        # Collect gradients corresponding to encoder layers
        for l in range(L):
            layer_base = getattr(base_model.vit.encoder.layer, str(l))
            layer_params = list(layer_base.parameters())
            for p, g in zip(layer_params, grads):
                if g is not None:
                    c_raw[l] += g.pow(2).sum()

    # Normalize FIM trace to get curvature scale
    c = c_raw / (c_raw.mean() + 1e-8)
    print("Normalized ViT-B/16 layer-wise curvatures:")
    for l in range(L):
        print(f"  Layer {l:2d} = {c[l]:.4f}")

    # Helper function to construct merged state dict for functional_call
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

    # 5. Evaluate accuracies of merged model
    def evaluate_model(coeffs):
        accs = []
        # Evaluate Expert 1 on Task 1 with merged backbone
        correct = 0
        total = 0
        with torch.no_grad():
            m_p1 = get_merged_params(coeffs, expert1)
            for pixel_values, labels in dl1:
                outputs = functional_call(expert1, m_p1, kwargs={"pixel_values": pixel_values.to(device)})
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels.to(device)).sum().item()
                total += labels.size(0)
        accs.append(correct / total)
        
        # Evaluate Expert 2 on Task 2 with merged backbone
        correct = 0
        total = 0
        with torch.no_grad():
            m_p2 = get_merged_params(coeffs, expert2)
            for pixel_values, labels in dl2:
                outputs = functional_call(expert2, m_p2, kwargs={"pixel_values": pixel_values.to(device)})
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels.to(device)).sum().item()
                total += labels.size(0)
        accs.append(correct / total)
        return accs

    # Initialize coefficients to Uniform (0.5, 0.5)
    coeffs_uniform = torch.ones((K, L), device=device) * 0.5
    uniform_accs = evaluate_model(coeffs_uniform)
    print(f"Uniform Baseline (0.5) - Task 1 Acc: {uniform_accs[0]*100:.2f}%, Task 2 Acc: {uniform_accs[1]*100:.2f}%, Avg: {np.mean(uniform_accs)*100:.2f}%")

    # 6. Test-Time Adaptation under Local Test-Time Noise
    print("\nRunning Test-Time Adaptation on Local Stream...")
    
    # Highly biased local stream containing Task 1 inputs
    tta_inputs = inputs_task1[:8].to(device)

    # Method A: Unconstrained AdaMerging
    print("Optimizing Unconstrained AdaMerging (no regularization)...")
    coeffs_ada = (torch.ones((K, L), device=device) * 0.5).requires_grad_(True)
    optimizer = torch.optim.Adam([coeffs_ada], lr=0.5)

    for step in range(20):
        optimizer.zero_grad()
        m_p1 = get_merged_params(coeffs_ada, expert1)
        outputs = functional_call(expert1, m_p1, kwargs={"pixel_values": tta_inputs})
        probs = torch.softmax(outputs.logits, dim=-1)
        loss_tta = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        
        loss_tta.backward()
        optimizer.step()
        
    ada_accs = evaluate_model(coeffs_ada.detach())
    print(f"Unconstrained AdaMerging final coeffs:\n{coeffs_ada.detach()}")
    print(f"Unconstrained AdaMerging - Task 1 Acc: {ada_accs[0]*100:.2f}%, Task 2 Acc: {ada_accs[1]*100:.2f}%, Avg: {np.mean(ada_accs)*100:.2f}%")

    # Method B: PolyMerge (d=2)
    print("\nOptimizing PolyMerge (d=2)...")
    poly_params = torch.zeros((K, 3), device=device, requires_grad=True)
    with torch.no_grad():
        poly_params[:, 0] = 0.5
    optimizer = torch.optim.Adam([poly_params], lr=0.1)

    for step in range(20):
        optimizer.zero_grad()
        coeffs_poly = torch.zeros((K, L), device=device)
        for l in range(L):
            x_l = l / (L - 1)
            coeffs_poly[:, l] = poly_params[:, 0] + poly_params[:, 1] * x_l + poly_params[:, 2] * (x_l ** 2)
            
        m_p1 = get_merged_params(coeffs_poly, expert1)
        outputs = functional_call(expert1, m_p1, kwargs={"pixel_values": tta_inputs})
        probs = torch.softmax(outputs.logits, dim=-1)
        loss_tta = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        
        loss_tta.backward()
        optimizer.step()
        
    final_coeffs_poly = torch.zeros((K, L), device=device)
    for l in range(L):
        x_l = l / (L - 1)
        final_coeffs_poly[:, l] = poly_params[:, 0] + poly_params[:, 1] * x_l + poly_params[:, 2] * (x_l ** 2)
        
    poly_accs = evaluate_model(final_coeffs_poly.detach())
    print(f"PolyMerge final coeffs:\n{final_coeffs_poly.detach()}")
    print(f"PolyMerge - Task 1 Acc: {poly_accs[0]*100:.2f}%, Task 2 Acc: {poly_accs[1]*100:.2f}%, Avg: {np.mean(poly_accs)*100:.2f}%")

    # Method C: Flat TV-Regularized AdaMerging
    print("\nOptimizing Flat TV-Regularized AdaMerging...")
    coeffs_tv = (torch.ones((K, L), device=device) * 0.5).requires_grad_(True)
    optimizer = torch.optim.Adam([coeffs_tv], lr=0.5)
    beta = 2.0

    for step in range(20):
        optimizer.zero_grad()
        m_p1 = get_merged_params(coeffs_tv, expert1)
        outputs = functional_call(expert1, m_p1, kwargs={"pixel_values": tta_inputs})
        probs = torch.softmax(outputs.logits, dim=-1)
        loss_tta = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        
        loss_reg = 0.0
        for l in range(1, L):
            loss_reg += ((coeffs_tv[:, l] - coeffs_tv[:, l-1])**2).sum()
        loss_reg += 0.1 * ((coeffs_tv - 0.5)**2).sum()
        
        loss_joint = loss_tta + beta * loss_reg
        loss_joint.backward()
        optimizer.step()

    tv_accs = evaluate_model(coeffs_tv.detach())
    print(f"Flat TV-Regularized final coeffs:\n{coeffs_tv.detach()}")
    print(f"Flat TV-Regularized - Task 1 Acc: {tv_accs[0]*100:.2f}%, Task 2 Acc: {tv_accs[1]*100:.2f}%, Avg: {np.mean(tv_accs)*100:.2f}%")

    # Method D: RCR-Merge (Ours)
    print("\nOptimizing RCR-Merge (with spatial curvature-weighted TV)...")
    coeffs_rcr = (torch.ones((K, L), device=device) * 0.5).requires_grad_(True)
    optimizer = torch.optim.Adam([coeffs_rcr], lr=0.5)
    beta = 2.0  # Regularization strength

    for step in range(20):
        optimizer.zero_grad()
        m_p1 = get_merged_params(coeffs_rcr, expert1)
        outputs = functional_call(expert1, m_p1, kwargs={"pixel_values": tta_inputs})
        probs = torch.softmax(outputs.logits, dim=-1)
        loss_tta = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        
        # Riemannian Curvature-Weighted TV regularizer + anchoring penalty
        loss_reg = 0.0
        for l in range(1, L):
            geom_mean = torch.sqrt(c[l] * c[l-1])
            loss_reg += geom_mean * ((coeffs_rcr[:, l] - coeffs_rcr[:, l-1])**2).sum()
        
        # Anchoring penalty
        loss_reg += 0.1 * ((coeffs_rcr - 0.5)**2).sum()
        
        loss_joint = loss_tta + beta * loss_reg
        loss_joint.backward()
        optimizer.step()

    rcr_accs = evaluate_model(coeffs_rcr.detach())
    print(f"RCR-Merge final coeffs:\n{coeffs_rcr.detach()}")
    print(f"RCR-Merge (Ours) - Task 1 Acc: {rcr_accs[0]*100:.2f}%, Task 2 Acc: {rcr_accs[1]*100:.2f}%, Avg: {np.mean(rcr_accs)*100:.2f}%")

    # Print summary table
    print("\n" + "="*60)
    print("ViT PILOT STUDY SUMMARY:")
    print("="*60)
    print(f"Method                     | Task 1 Acc | Task 2 Acc | Average Acc")
    print(f"Uniform Baseline (0.5)     | {uniform_accs[0]*100:9.2f}% | {uniform_accs[1]*100:9.2f}% | {np.mean(uniform_accs)*100:9.2f}%")
    print(f"Unconstrained AdaMerging   | {ada_accs[0]*100:9.2f}% | {ada_accs[1]*100:9.2f}% | {np.mean(ada_accs)*100:9.2f}%")
    print(f"PolyMerge (d=2)            | {poly_accs[0]*100:9.2f}% | {poly_accs[1]*100:9.2f}% | {np.mean(poly_accs)*100:9.2f}%")
    print(f"Flat TV-Regularized        | {tv_accs[0]*100:9.2f}% | {tv_accs[1]*100:9.2f}% | {np.mean(tv_accs)*100:9.2f}%")
    print(f"RCR-Merge (Ours)           | {rcr_accs[0]*100:9.2f}% | {rcr_accs[1]*100:9.2f}% | {np.mean(rcr_accs)*100:9.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
