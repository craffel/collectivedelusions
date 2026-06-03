import os
import random
import numpy as np
import torch
import torch.nn as nn
from run_experiments import (
    get_dataloaders, ExpertModelResNet, apply_de_bn, 
    compute_weight_averaging, evaluate_model, device
)

def run_robustness_analysis():
    print(f"Using device: {device}")
    
    # Load dataloaders
    loaders = get_dataloaders()
    tasks = ["mnist", "fmnist", "cifar"]
    
    # Initialize progenitor
    progenitor = ExpertModelResNet().to(device)
    progenitor_path = "resnet_progenitor.pt"
    if os.path.exists(progenitor_path):
        print(f"Loading progenitor from {progenitor_path}...")
        progenitor.backbone.load_state_dict(torch.load(progenitor_path, map_location=device))
    progenitor_state = progenitor.backbone.state_dict()
    
    # Load experts
    expert_backbones = []
    expert_heads = []
    
    for i, task in enumerate(tasks):
        expert_path = f"resnet_expert_{task}.pt"
        if os.path.exists(expert_path):
            print(f"Loading expert from {expert_path}...")
            checkpoint = torch.load(expert_path, map_location=device)
            expert_backbones.append(checkpoint["backbone"])
            expert_heads.append(checkpoint["fc"])
        else:
            raise FileNotFoundError(f"Expert model checkpoint {expert_path} not found!")
            
    # Compute Weight Averaging
    print("Computing Weight Averaging (WA) backbone...")
    wa_backbone = compute_weight_averaging(progenitor_state, expert_backbones)
    
    # Define sample sizes and seeds
    sample_sizes = [8, 16, 32, 64]
    seeds = [42, 100, 2026, 999, 12345]
    
    robustness_results = {}
    
    for N in sample_sizes:
        print(f"\nEvaluating DE-BN (WA, N={N}) across {len(seeds)} random seeds...")
        seed_accs = []
        
        for seed in seeds:
            # Set seed for both python/numpy/pytorch and standard random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
            task_accs = {}
            for i, task in enumerate(tasks):
                test_model = ExpertModelResNet().to(device)
                test_model.backbone.load_state_dict(wa_backbone)
                test_model.fc.load_state_dict(expert_heads[i])
                
                # Apply DE-BN calibration
                test_model = apply_de_bn(test_model, loaders[task]["raw_train"], N)
                
                # Evaluate
                acc = evaluate_model(test_model, loaders[task]["test"])
                task_accs[task] = acc
                
            avg_acc = sum(task_accs.values()) / len(task_accs)
            seed_accs.append(avg_acc)
            print(f"  Seed {seed:5d} -> Avg Acc: {avg_acc:.2f}% (MNIST: {task_accs['mnist']:.2f}%, FMNIST: {task_accs['fmnist']:.2f}%, CIFAR: {task_accs['cifar']:.2f}%)")
            
        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        robustness_results[N] = {
            "all_accs": seed_accs,
            "mean": mean_acc,
            "std": std_acc
        }
        print(f"==> N={N:2d} Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
    # Save results
    torch.save(robustness_results, "robustness_results.pt")
    print("\nRobustness analysis completed and saved to robustness_results.pt!")

if __name__ == "__main__":
    run_robustness_analysis()
