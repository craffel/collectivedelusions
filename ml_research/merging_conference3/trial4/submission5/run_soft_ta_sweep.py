import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from train_and_merge import (
    set_seed, load_task_datasets, prepare_dataloaders,
    get_task_vector, evaluate_merged_model, construct_merged_backbone
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def apply_soft_sg_ta_masking(task_vectors, base_state, keep_ratio, beta=10.0, masking_type="GQ"):
    """
    Applies sigmoid-gated (soft) masking to task vectors.
    v_gated = v * sigmoid(beta * (|v|/threshold - 1.0))
    """
    gated_task_vectors = {}
    
    for task_name, tv in task_vectors.items():
        gated_task_vectors[task_name] = {}
        
        if masking_type == "GQ":
            # Global Soft Quantile
            all_vals = []
            for k, v in tv.items():
                all_vals.append(v.flatten())
            all_vals_tensor = torch.cat(all_vals)
            num_keep = int(keep_ratio * len(all_vals_tensor))
            if num_keep == 0:
                num_keep = 1
            threshold = torch.topk(torch.abs(all_vals_tensor), num_keep).values[-1]
            # Ensure threshold is non-zero to avoid division by zero
            threshold = max(threshold.item(), 1e-12)
            
            for k, v in tv.items():
                ratio = torch.abs(v) / threshold
                gate = torch.sigmoid(beta * (ratio - 1.0))
                gated_task_vectors[task_name][k] = gate * v
                
        elif masking_type == "LQ":
            # Layer-wise Soft Quantile
            for k, v in tv.items():
                flat_v = v.flatten()
                num_keep = int(keep_ratio * len(flat_v))
                if num_keep == 0:
                    num_keep = 1
                threshold = torch.topk(torch.abs(flat_v), num_keep).values[-1]
                threshold = max(threshold.item(), 1e-12)
                
                ratio = torch.abs(v) / threshold
                gate = torch.sigmoid(beta * (ratio - 1.0))
                gated_task_vectors[task_name][k] = gate * v
                
    return gated_task_vectors

def main():
    set_seed(42)
    datasets = load_task_datasets()
    task_names = list(datasets.keys())
    
    # Load base model
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
    
    # Load expert states
    expert_states = {}
    expert_heads = {}
    task_vectors = {}
    for task_name in task_names:
        path = f"./checkpoints/{task_name}_expert.pt"
        state = torch.load(path, map_location="cpu")
        expert_states[task_name] = state
        
        # Extract heads
        expert_heads[task_name] = {}
        for k, v in state.items():
            if k.startswith("head."):
                head_key = k.replace("head.", "")
                expert_heads[task_name][head_key] = v.clone()
                
        # Get task vector
        task_vectors[task_name] = get_task_vector(state, base_state)
        
    print("Loaded experts and task vectors.")
    
    seeds = [42, 100, 2026, 777, 999]
    keep_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    betas = [5.0, 10.0, 20.0]
    
    results = {"GQ-Soft": [], "LQ-Soft": []}
    
    for s_idx, seed in enumerate(seeds):
        print(f"\n=================== RUNNING SEED {seed} ({s_idx+1}/{len(seeds)}) ===================")
        train_loaders, test_loaders, val_data = prepare_dataloaders(datasets, seed)
        
        for masking_type in ["GQ", "LQ"]:
            method_key = f"{masking_type}-Soft"
            best_val_acc = -1
            best_k = 0.5
            best_alpha = 0.7
            best_beta = 10.0
            
            # Sweep parameters on validation subset
            for k in keep_ratios:
                for beta in betas:
                    # Short-circuit: if k=1.0, threshold does not matter, gate becomes ~sigmoid(beta*(|v|/1e-12 - 1)) which is 1
                    # So we only need to test one beta for k=1.0
                    if k == 1.0 and beta != betas[0]:
                        continue
                        
                    masked_tvs = apply_soft_sg_ta_masking(task_vectors, base_state, k, beta, masking_type)
                    for alpha in alphas:
                        weights = {task: alpha for task in task_names}
                        backbone = construct_merged_backbone(base_state, masked_tvs, weights)
                        val_accs = evaluate_merged_model(backbone, expert_heads, test_loaders, val_data)
                        if val_accs["Joint Mean"] > best_val_acc:
                            best_val_acc = val_accs["Joint Mean"]
                            best_k = k
                            best_alpha = alpha
                            best_beta = beta
            
            # Evaluate optimal parameters on the full test sets
            print(f"Optimal parameters for {method_key} on seed {seed}: k={best_k:.2f}, beta={best_beta:.2f}, alpha={best_alpha:.2f}. Val Joint Acc: {best_val_acc:.2f}%")
            opt_masked_tvs = apply_soft_sg_ta_masking(task_vectors, base_state, best_k, best_beta, masking_type)
            opt_weights = {task: best_alpha for task in task_names}
            opt_backbone = construct_merged_backbone(base_state, opt_masked_tvs, opt_weights)
            test_accs = evaluate_merged_model(opt_backbone, expert_heads, test_loaders, None)
            results[method_key].append({
                "Joint Mean": test_accs["Joint Mean"],
                "MNIST": test_accs["MNIST"],
                "FashionMNIST": test_accs["FashionMNIST"],
                "CIFAR10": test_accs["CIFAR10"],
                "SVHN": test_accs["SVHN"],
                "k": best_k,
                "beta": best_beta,
                "alpha": best_alpha
            })
            print(f"Seed {seed} {method_key} Test Acc: {test_accs['Joint Mean']:.2f}%")
            
    # Print and save aggregate results
    print("\n" + "="*30 + " AGGREGATE SOFT-TA RESULTS ACROSS 5 SEEDS " + "="*30)
    aggregate_results = {}
    for method_key, seed_res in results.items():
        joint_accs = [r["Joint Mean"] for r in seed_res]
        mean_joint = np.mean(joint_accs)
        std_joint = np.std(joint_accs)
        
        # Compute mean per dataset
        mean_ds = {}
        for d in task_names:
            mean_ds[d] = np.mean([r[d] for r in seed_res])
            
        mean_k = np.mean([r["k"] for r in seed_res])
        mean_beta = np.mean([r["beta"] for r in seed_res])
        mean_alpha = np.mean([r["alpha"] for r in seed_res])
        
        aggregate_results[method_key] = {
            "Joint Mean": mean_joint,
            "Joint Std": std_joint,
            "MNIST": mean_ds["MNIST"],
            "FashionMNIST": mean_ds["FashionMNIST"],
            "CIFAR10": mean_ds["CIFAR10"],
            "SVHN": mean_ds["SVHN"],
            "mean_k": mean_k,
            "mean_beta": mean_beta,
            "mean_alpha": mean_alpha
        }
        print(f"{method_key:<10} Joint Mean Accuracy: {mean_joint:.2f}% ± {std_joint:.2f}% (k={mean_k:.2f}, beta={mean_beta:.1f}, alpha={mean_alpha:.2f})")
        
    with open("./results/soft_ta_metrics.json", "w") as f:
        json.dump(aggregate_results, f, indent=2)
    print("Saved soft-ta results to ./results/soft_ta_metrics.json")

if __name__ == "__main__":
    main()
