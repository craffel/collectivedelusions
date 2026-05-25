import torch
import torch.nn as nn
import numpy as np
from evaluate_ttmm import ExpertCNN, get_merged_state_dict, compute_batch_cohesion

def test_bn_buffer_merging():
    print("Testing Batch Normalization Buffer Merging...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create mock experts
    expert1 = ExpertCNN().to(device)
    expert2 = ExpertCNN().to(device)
    expert3 = ExpertCNN().to(device)
    
    # Set different running stats in expert BN layers
    with torch.no_grad():
        expert1.bn1.running_mean.fill_(1.0)
        expert1.bn1.running_var.fill_(2.0)
        
        expert2.bn1.running_mean.fill_(3.0)
        expert2.bn1.running_var.fill_(4.0)
        
        expert3.bn1.running_mean.fill_(5.0)
        expert3.bn1.running_var.fill_(6.0)
        
    experts_list = [expert1, expert2, expert3]
    base_model = ExpertCNN().to(device)
    
    # Define coefficients (softmax weights of [0.5, 0.3, 0.2])
    # logits: [0.0, -0.5108, -0.9163]
    coeffs = {}
    for name, param in base_model.named_parameters():
        coeffs[name] = torch.tensor([0.0, -0.51082562, -0.91629073], device=device)
        
    # Merge with BN = True
    merged_sd_with_bn = get_merged_state_dict(experts_list, coeffs, base_model, use_bn=True)
    
    # Expected mean: 0.5 * 1.0 + 0.3 * 3.0 + 0.2 * 5.0 = 0.5 + 0.9 + 1.0 = 2.4
    # Expected var: 0.5 * 2.0 + 0.3 * 4.0 + 0.2 * 6.0 = 1.0 + 1.2 + 1.2 = 3.4
    
    merged_mean = merged_sd_with_bn["bn1.running_mean"][0].item()
    merged_var = merged_sd_with_bn["bn1.running_var"][0].item()
    
    print(f"Merged BN mean: {merged_mean:.4f} (Expected: 2.4000)")
    print(f"Merged BN var:  {merged_var:.4f} (Expected: 3.4000)")
    
    assert abs(merged_mean - 2.4) < 1e-3, f"BN mean mismatch: {merged_mean}"
    assert abs(merged_var - 3.4) < 1e-3, f"BN var mismatch: {merged_var}"
    print("BN Buffer Merging Test PASSED.\n")

def test_tt_fisher_computation():
    print("Testing Test-Time Fisher preconditioning calculation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ExpertCNN().to(device)
    inputs = torch.randn(8, 1, 28, 28, device=device)
    
    # Compute output and pseudo-labels
    outputs = model(inputs)
    pseudo_labels = torch.argmax(outputs, dim=1)
    
    # Compute TT-Fisher sensitivities
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, pseudo_labels)
    
    model.zero_grad()
    loss.backward()
    
    grad_scales = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sq_mean = param.grad.pow(2).mean().item()
                grad_scales[name] = 1.0 / (grad_sq_mean + 1e-4)**0.5
                
    print("Preconditioning scales calculated successfully:")
    for name, scale in list(grad_scales.items())[:3]:
        print(f"  {name:15s} -> preconditioning scale: {scale:.4f}")
        
    assert len(grad_scales) > 0, "No gradients calculated"
    print("TT-Fisher Computation Test PASSED.\n")

def test_cohesion_computation():
    print("Testing Batch Cohesion score computation...")
    # Mock features and prototypes as a dictionary from class (0-9) to prototype tensors
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]])
    prototypes = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0])
    }
    for c in range(2, 10):
        prototypes[c] = torch.tensor([0.0, 0.0])
        
    mu_static = torch.tensor([0.0, 0.0])
    
    # Max cosine similarities:
    # row 0: max(cos(feats[0], prototypes)) = max(1.0, 0.0) = 1.0
    # row 1: max(cos(feats[1], prototypes)) = max(0.0, 1.0) = 1.0
    # row 2: max(cos(feats[2], prototypes)) = max(0.707, 0.707) = 0.707
    # Avg: (1.0 + 1.0 + 0.707) / 3 = 0.9023
    
    cohesion = compute_batch_cohesion(feats, prototypes, mu_static)
    print(f"Computed cohesion: {cohesion:.4f} (Expected: 0.9023)")
    assert abs(cohesion - 0.9023) < 1e-3, f"Cohesion mismatch: {cohesion}"
    print("Cohesion Computation Test PASSED.\n")

if __name__ == "__main__":
    test_bn_buffer_merging()
    test_tt_fisher_computation()
    test_cohesion_computation()
    print("All Unit Tests Passed Successfully!")
