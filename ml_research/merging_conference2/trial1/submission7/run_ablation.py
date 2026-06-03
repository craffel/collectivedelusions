import os
import json
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader, Subset
from eval_and_tta import get_dataloaders, evaluate_model, perform_tta

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Ablation (Perturbation Geometry) on {device}")
    
    loader_c10, loader_svhn, test_c10, test_svhn = get_dataloaders()
    
    # Initialize base backbone
    backbone_base = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device)
    base_state = {k: v.to(device) for k, v in backbone_base.state_dict().items()}
    
    seeds = [42, 43, 44]
    shot_list = [4, 16, 64, 256]
    
    # Define ablation methods to run:
    # (name, mode, opt_method, lr, steps, rho, loss_type)
    ablation_methods = [
        ("SA-TTA_Sign_KL", "global", "sam_sign", 0.05, 40, 0.1, "kl_expert"),
        ("SA-TTA_Layerwise_Sign_KL", "layerwise", "sam_sign", 0.01, 30, 0.1, "kl_expert")
    ]
    
    # Pre-load checkpoints
    checkpoints = {}
    for seed in seeds:
        c10_path = f"./checkpoints/cifar10_seed{seed}.pth"
        svhn_path = f"./checkpoints/svhn_seed{seed}.pth"
        if os.path.exists(c10_path) and os.path.exists(svhn_path):
            checkpoint_A = torch.load(c10_path, map_location=device)
            checkpoint_B = torch.load(svhn_path, map_location=device)
            checkpoints[seed] = (checkpoint_A, checkpoint_B)
            
    print("Checkpoints loaded successfully.")
    
    ablation_results = {m[0]: {str(shot): [] for shot in shot_list} for m in ablation_methods}
    
    for shot in shot_list:
        print(f"\nEvaluating {shot} shots...")
        indices = list(range(shot))
        subset_c10 = Subset(test_c10, indices)
        subset_svhn = Subset(test_svhn, indices)
        loader_sub_c10 = DataLoader(subset_c10, batch_size=shot, shuffle=False)
        loader_sub_svhn = DataLoader(subset_svhn, batch_size=shot, shuffle=False)
        x_c10, _ = next(iter(loader_sub_c10))
        x_svhn, _ = next(iter(loader_sub_svhn))
        x_c10, x_svhn = x_c10.to(device), x_svhn.to(device)
        
        for seed in seeds:
            if seed not in checkpoints:
                continue
            checkpoint_A, checkpoint_B = checkpoints[seed]
            
            expert_A_state = checkpoint_A['backbone_state_dict']
            expert_B_state = checkpoint_B['backbone_state_dict']
            
            head_c10 = nn.Linear(512, 10).to(device)
            head_svhn = nn.Linear(512, 10).to(device)
            head_c10.load_state_dict(checkpoint_A['head_state_dict'])
            head_svhn.load_state_dict(checkpoint_B['head_state_dict'])
            
            delta_A = {k: expert_A_state[k].to(device) - base_state[k] for k in base_state}
            delta_B = {k: expert_B_state[k].to(device) - base_state[k] for k in base_state}
            
            # Expert A and B params for precomputing logits
            params_expert_A = {k: base_state[k] + 1.0 * delta_A[k] for k in base_state}
            params_expert_B = {k: base_state[k] + 1.0 * delta_B[k] for k in base_state}
            
            with torch.no_grad():
                features_A = torch.func.functional_call(backbone_base, params_expert_A, x_c10)
                expert_logits_c10 = head_c10(features_A)
                
                features_B = torch.func.functional_call(backbone_base, params_expert_B, x_svhn)
                expert_logits_svhn = head_svhn(features_B)
                
            for name, mode, opt_method, lr, steps, rho, loss_type in ablation_methods:
                print(f"Running {name} on Seed {seed}...")
                adapted_params = perform_tta(
                    base_state, delta_A, delta_B, backbone_base, head_c10, head_svhn,
                    x_c10, x_svhn, expert_logits_c10, expert_logits_svhn,
                    lr=lr, steps=steps, mode=mode, opt_method=opt_method, rho=rho, loss_type=loss_type, device=device
                )
                c10_acc, svhn_acc, avg_acc = evaluate_model(adapted_params, backbone_base, head_c10, head_svhn, loader_c10, loader_svhn, device)
                print(f"Result {name} -> Avg: {avg_acc:.2f}%")
                ablation_results[name][str(shot)].append(avg_acc)
                
    # Save raw ablation results
    with open("ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
        print("Ablation results saved to ablation_results.json")
        
    # Generate summaries
    print("\n==========================================")
    print("ABLATION RESULTS SUMMARY (MEAN +- STD ERR)")
    print("==========================================")
    for name in ablation_results:
        print(f"\nMethod: {name}")
        for shot in shot_list:
            vals = ablation_results[name][str(shot)]
            mean = np.mean(vals)
            stderr = np.std(vals) / np.sqrt(len(vals))
            print(f"  {shot} shots: {mean:.2f} +- {stderr:.2f}%")

if __name__ == "__main__":
    main()
