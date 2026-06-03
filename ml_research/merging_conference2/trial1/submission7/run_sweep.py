import os
import json
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader, Subset
from eval_and_tta import get_dataloaders, evaluate_model, perform_tta, entropy_loss, kl_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Sweep on {device}")
    
    loader_c10, loader_svhn, test_c10, test_svhn = get_dataloaders()
    
    # Subsets for extremely fast evaluation during sweep (1000 and 2000 samples)
    eval_subset_c10 = Subset(test_c10, list(range(1000)))
    eval_subset_svhn = Subset(test_svhn, list(range(2000)))
    loader_c10_eval = DataLoader(eval_subset_c10, batch_size=256, shuffle=False)
    loader_svhn_eval = DataLoader(eval_subset_svhn, batch_size=256, shuffle=False)
    
    # Initialize base backbone
    backbone_base = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device)
    base_state = {k: v.to(device) for k, v in backbone_base.state_dict().items()}
    
    seeds = [42, 43, 44]
    shot_list = [4, 16, 64, 256]
    
    # 1. Sweep over rho
    rho_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    rho_results = {}
    
    # 2. Sweep over lr
    lr_values = [0.005, 0.01, 0.05, 0.1]
    lr_results = {}
    
    # Pre-load checkpoints and heads to save time
    checkpoints = {}
    for seed in seeds:
        c10_path = f"./checkpoints/cifar10_seed{seed}.pth"
        svhn_path = f"./checkpoints/svhn_seed{seed}.pth"
        
        if os.path.exists(c10_path) and os.path.exists(svhn_path):
            checkpoint_A = torch.load(c10_path, map_location=device)
            checkpoint_B = torch.load(svhn_path, map_location=device)
            checkpoints[seed] = (checkpoint_A, checkpoint_B)
            
    print("Checkpoints loaded successfully.")
    
    # Run rho sweep
    print("\n--- STARTING RHO SWEEP ---")
    for rho in rho_values:
        print(f"Sweeping rho = {rho}")
        rho_results[str(rho)] = {}
        for shot in shot_list:
            rho_results[str(rho)][str(shot)] = []
            
            # Prepare subsets for each shot
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
                
                # Evaluate SA-TTA KL with given rho
                opt_method = "sam" if rho > 0 else "standard"
                adapted_params = perform_tta(
                    base_state, delta_A, delta_B, backbone_base, head_c10, head_svhn,
                    x_c10, x_svhn, expert_logits_c10, expert_logits_svhn,
                    lr=0.05, steps=40, mode="global", opt_method=opt_method, rho=rho, loss_type="kl_expert", device=device
                )
                
                c10_acc, svhn_acc, avg_acc = evaluate_model(adapted_params, backbone_base, head_c10, head_svhn, loader_c10_eval, loader_svhn_eval, device)
                rho_results[str(rho)][str(shot)].append(avg_acc)
                
    # Run lr sweep
    print("\n--- STARTING LR SWEEP ---")
    for lr in lr_values:
        print(f"Sweeping lr = {lr}")
        lr_results[str(lr)] = {}
        for shot in shot_list:
            lr_results[str(lr)][str(shot)] = []
            
            # Prepare subsets for each shot
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
                
                # Evaluate SA-TTA KL with given lr and fixed rho=0.1
                adapted_params = perform_tta(
                    base_state, delta_A, delta_B, backbone_base, head_c10, head_svhn,
                    x_c10, x_svhn, expert_logits_c10, expert_logits_svhn,
                    lr=lr, steps=40, mode="global", opt_method="sam", rho=0.1, loss_type="kl_expert", device=device
                )
                
                c10_acc, svhn_acc, avg_acc = evaluate_model(adapted_params, backbone_base, head_c10, head_svhn, loader_c10_eval, loader_svhn_eval, device)
                lr_results[str(lr)][str(shot)].append(avg_acc)

    # Save sweep results
    with open("sweep_results.json", "w") as f:
        json.dump({
            "rho_sweep": rho_results,
            "lr_sweep": lr_results
        }, f, indent=4)
        print("\nSweep results saved to sweep_results.json")

if __name__ == "__main__":
    main()
