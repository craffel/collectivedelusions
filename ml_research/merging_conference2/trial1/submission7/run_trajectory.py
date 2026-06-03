import os
import json
import torch
import torch.nn as nn
import numpy as np
import timm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from eval_and_tta import get_dataloaders, evaluate_model, perform_tta

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Trajectory Analysis on {device}")
    
    loader_c10, loader_svhn, test_c10, test_svhn = get_dataloaders()
    
    # Initialize base backbone
    backbone_base = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device)
    base_state = {k: v.to(device) for k, v in backbone_base.state_dict().items()}
    
    seeds = [42, 43, 44]
    # We will analyze 256 shots because that is where standard entropy collapses and KL converges
    shots = 256
    step_intervals = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    
    methods = [
        # (name, mode, opt_method, lr, rho, loss_type)
        ("AdaMerging_Entropy", "global", "standard", 0.05, 0.0, "entropy"),
        ("SA-TTA_Entropy", "global", "sam", 0.05, 0.1, "entropy"),
        ("AdaMerging_KL", "global", "standard", 0.05, 0.0, "kl_expert"),
        ("SA-TTA_KL", "global", "sam", 0.05, 0.1, "kl_expert"),
    ]
    
    # Load checkpoints and experts
    checkpoints = {}
    for seed in seeds:
        c10_path = f"./checkpoints/cifar10_seed{seed}.pth"
        svhn_path = f"./checkpoints/svhn_seed{seed}.pth"
        
        if os.path.exists(c10_path) and os.path.exists(svhn_path):
            checkpoint_A = torch.load(c10_path, map_location=device)
            checkpoint_B = torch.load(svhn_path, map_location=device)
            checkpoints[seed] = (checkpoint_A, checkpoint_B)
            
    print("Checkpoints loaded successfully.")
    
    trajectory_data = {m[0]: {str(step): [] for step in step_intervals} for m in methods}
    
    # Extract the test-time adaptation shots
    indices = list(range(shots))
    subset_c10 = Subset(test_c10, indices)
    subset_svhn = Subset(test_svhn, indices)
    
    loader_sub_c10 = DataLoader(subset_c10, batch_size=shots, shuffle=False)
    loader_sub_svhn = DataLoader(subset_svhn, batch_size=shots, shuffle=False)
    
    x_c10, _ = next(iter(loader_sub_c10))
    x_svhn, _ = next(iter(loader_sub_svhn))
    x_c10 = x_c10.to(device)
    x_svhn = x_svhn.to(device)
    
    for seed in seeds:
        if seed not in checkpoints:
            continue
        print(f"\nEvaluating Seed {seed}...")
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
            
        # Step 0 is the same for all: uniform merge
        params_uniform = {k: base_state[k] + 0.5 * delta_A[k] + 0.5 * delta_B[k] for k in base_state}
        uni_c10, uni_svhn, uni_avg = evaluate_model(params_uniform, backbone_base, head_c10, head_svhn, loader_c10, loader_svhn, device)
        print(f"Step 0 Uniform Merge -> CIFAR-10 Acc: {uni_c10:.2f}%, SVHN Acc: {uni_svhn:.2f}% (Avg: {uni_avg:.2f}%)")
        
        for m_name, mode, opt_method, lr, rho, loss_type in methods:
            trajectory_data[m_name]["0"].append(uni_avg)
            
            for step in step_intervals:
                if step == 0:
                    continue
                print(f"Running {m_name} for {step} steps...")
                adapted_params = perform_tta(
                    base_state, delta_A, delta_B, backbone_base, head_c10, head_svhn,
                    x_c10, x_svhn, expert_logits_c10, expert_logits_svhn,
                    lr=lr, steps=step, mode=mode, opt_method=opt_method, rho=rho, loss_type=loss_type, device=device
                )
                c10_acc, svhn_acc, avg_acc = evaluate_model(adapted_params, backbone_base, head_c10, head_svhn, loader_c10, loader_svhn, device)
                trajectory_data[m_name][str(step)].append(avg_acc)
                print(f"Result {m_name} at step {step} -> Avg: {avg_acc:.2f}%")
                
    # Compute mean and standard error for plotting
    plot_mean = {m[0]: [] for m in methods}
    plot_err = {m[0]: [] for m in methods}
    
    for m_name in trajectory_data:
        for step in step_intervals:
            vals = trajectory_data[m_name][str(step)]
            plot_mean[m_name].append(np.mean(vals))
            plot_err[m_name].append(np.std(vals) / np.sqrt(len(seeds)))
            
    # Plotting
    plt.figure(figsize=(10, 6))
    colors = {
        "AdaMerging_Entropy": "red",
        "SA-TTA_Entropy": "darkred",
        "AdaMerging_KL": "blue",
        "SA-TTA_KL": "darkblue",
    }
    markers = {
        "AdaMerging_Entropy": "o",
        "SA-TTA_Entropy": "s",
        "AdaMerging_KL": "o",
        "SA-TTA_KL": "s",
    }
    
    for m_name in trajectory_data:
        plt.errorbar(
            step_intervals, plot_mean[m_name], yerr=plot_err[m_name],
            label=m_name.replace("_", " "), color=colors[m_name], marker=markers[m_name], linestyle="-", capsize=5
        )
        
    plt.xlabel("Adaptation Steps", fontsize=12)
    plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=12)
    plt.title("TTA Convergence and Stability Trajectory (256 Shots)", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    plt.savefig("tta_trajectory_analysis.png", dpi=300)
    print("Trajectory plot saved to tta_trajectory_analysis.png")
    
    # Save raw trajectory results
    with open("trajectory_results.json", "w") as f:
        json.dump(trajectory_data, f, indent=4)
        print("Trajectory results saved to trajectory_results.json")

if __name__ == "__main__":
    main()
