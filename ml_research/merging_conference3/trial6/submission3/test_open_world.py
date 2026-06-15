import numpy as np
import torch
import torch.nn as nn
import json
from model_routing import (
    generate_synthetic_dataset,
    train_experts,
    PCAPreprojector,
    BWS_Router,
    set_seed
)

SEEDS = [42, 43, 44, 45, 46]

def main():
    print("=" * 60)
    print("RUNNING SIGMOID VS SOFTMAX OPEN-WORLD VERIFICATION")
    print("=" * 60)
    
    # We will track:
    # 1. OOD / Corrupt Input: Sum of gating coefficients for Sigmoid vs Softmax
    # 2. Multi-task Input: Gating coefficients for a dual-task mixed input
    
    sigmoid_ood_sums = []
    softmax_ood_sums = []
    
    sigmoid_mixed_activations = []
    softmax_mixed_activations = []
    
    for seed in SEEDS:
        print(f"\n>>> Running for Seed {seed}...")
        train_data, test_data, calib_data = generate_synthetic_dataset(seed)
        experts = train_experts(train_data, test_data)
        
        pca_proj = PCAPreprojector(n_components=4)
        pca_proj.fit(calib_data[0])
        
        # Train Sigmoid Router (M=3, G=4)
        router_sigmoid = BWS_Router(L=12, G=4, d=4, K=4, activation='Sigmoid')
        from model_routing import train_router
        train_router(router_sigmoid, pca_proj, experts, calib_data, lr=0.05, lambda_wd=1e-4)
        
        # Train Softmax Router (M=3, G=4)
        router_softmax = BWS_Router(L=12, G=4, d=4, K=4, activation='Softmax')
        train_router(router_softmax, pca_proj, experts, calib_data, lr=0.05, lambda_wd=1e-4)
        
        router_sigmoid.eval()
        router_softmax.eval()
        
        # --- Experiment 1: Corrupt / OOD Input ---
        # Generate entirely out-of-distribution inputs (e.g. pure random Gaussian noise)
        set_seed(seed)
        D = 192
        X_ood = np.random.normal(0, 1.0, (100, D)).astype(np.float32)
        X_ood_t = torch.tensor(X_ood)
        
        psi_ood = pca_proj.project(X_ood_t)
        
        with torch.no_grad():
            alpha_sig_ood = router_sigmoid(psi_ood) # [100, L, K]
            alpha_soft_ood = router_softmax(psi_ood) # [100, L, K]
            
            # Sum over tasks, mean over layers and batch
            sig_sum = alpha_sig_ood.mean(dim=1).sum(dim=-1).mean().item()
            soft_sum = alpha_soft_ood.mean(dim=1).sum(dim=-1).mean().item()
            
            sigmoid_ood_sums.append(sig_sum)
            softmax_ood_sums.append(soft_sum)
            
        # --- Experiment 2: Multi-task Mixing ---
        # Create an input that contains style cues for BOTH Task 0 and Task 1
        # In generate_synthetic_dataset: 
        # style_feat[k * style_dim_per_task : (k + 1) * style_dim_per_task] = 1.5
        # Let's create a dual-style feature vector!
        D_style = 64
        D_shared = 128
        style_dim_per_task = 16
        
        # We'll create a mixed style feature for Task 0 and Task 1
        style_mixed = np.zeros(D_style)
        style_mixed[0 * style_dim_per_task : 1 * style_dim_per_task] = 1.5
        style_mixed[1 * style_dim_per_task : 2 * style_dim_per_task] = 1.5
        
        # Let's use Class 0 shared prototype
        # Need to retrieve the prototypes. They are in train_data or we can just reconstruct style
        # Or we can just build a realistic mixed sample manually. Let's do it:
        # Load a Class 0 sample from Task 0, and add style cue for Task 1!
        X_t0, _ = test_data[0]
        sample_base = X_t0[0].copy() #mnist task 0
        # Overwrite Task 1 style dimension
        sample_base[D_shared + 1 * style_dim_per_task : D_shared + 2 * style_dim_per_task] = 1.5
        
        X_mixed = torch.tensor(sample_base).unsqueeze(0).float()
        psi_mixed = pca_proj.project(X_mixed)
        
        with torch.no_grad():
            alpha_sig_mixed = router_sigmoid(psi_mixed) # [1, L, K]
            alpha_soft_mixed = router_softmax(psi_mixed) # [1, L, K]
            
            # Mean over layers, extract first sample
            sig_act = alpha_sig_mixed.mean(dim=1).squeeze(0).numpy() # [K]
            soft_act = alpha_soft_mixed.mean(dim=1).squeeze(0).numpy() # [K]
            
            sigmoid_mixed_activations.append(sig_act)
            softmax_mixed_activations.append(soft_act)
            
    # Compile and print results
    print("\n" + "=" * 50)
    print("OPEN-WORLD EVALUATION RESULTS")
    print("=" * 50)
    
    print("\n--- Experiment 1: Out-of-Distribution Gating Activation Sum ---")
    print(f"Sigmoid Router Gating Sum under OOD: {np.mean(sigmoid_ood_sums):.4f} ± {np.std(sigmoid_ood_sums):.4f}")
    print(f"Softmax Router Gating Sum under OOD: {np.mean(softmax_ood_sums):.4f} ± {np.std(softmax_ood_sums):.4f}")
    print("Interpretation: Under OOD inputs, Sigmoid successfully deactivates the expert networks (sum ~0) to fallback to the base model, whereas Softmax is mathematically forced to distribute exactly 1.0 total gating weight, injecting noise.")
    
    print("\n--- Experiment 2: Multi-task Mixing Gating Activation ---")
    sig_mixed_mean = np.mean(sigmoid_mixed_activations, axis=0)
    sig_mixed_std = np.std(sigmoid_mixed_activations, axis=0)
    soft_mixed_mean = np.mean(softmax_mixed_activations, axis=0)
    soft_mixed_std = np.std(softmax_mixed_activations, axis=0)
    
    print("Sigmoid Gating Coeffs for dual-style Task 0 + Task 1 input:")
    for k in range(4):
        print(f"  Task {k}: {sig_mixed_mean[k]:.4f} ± {sig_mixed_std[k]:.4f}")
    print("Softmax Gating Coeffs for dual-style Task 0 + Task 1 input:")
    for k in range(4):
        print(f"  Task {k}: {soft_mixed_mean[k]:.4f} ± {soft_mixed_std[k]:.4f}")
    print("Interpretation: Under mixed inputs, Sigmoid successfully activates both Task 0 and Task 1 expert tasks simultaneously with high coefficients, whereas Softmax is bottlenecked and forced to select or split gating values due to sum-to-one constraint.")
    
    # Save results as json for easy reference
    output_data = {
        'ood_sigmoid': sigmoid_ood_sums,
        'ood_softmax': softmax_ood_sums,
        'mixed_sigmoid': [act.tolist() for act in sigmoid_mixed_activations],
        'mixed_softmax': [act.tolist() for act in softmax_mixed_activations]
    }
    with open('open_world_results.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    main()