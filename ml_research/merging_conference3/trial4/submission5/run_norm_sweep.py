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
    get_task_vector, evaluate_merged_model, apply_sg_ta_masking,
    construct_merged_backbone
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    
    # Compute mean absolute magnitudes
    mean_abs_mags = {}
    for task_name, tv in task_vectors.items():
        total_abs_sum = 0.0
        total_numel = 0
        for k, v in tv.items():
            total_abs_sum += torch.sum(torch.abs(v)).item()
            total_numel += v.numel()
        mean_abs_mags[task_name] = total_abs_sum / total_numel
        print(f"Task {task_name}: mean abs mag = {mean_abs_mags[task_name]:.6f}")
        
    # Create normalized task vectors
    norm_task_vectors = {}
    for task_name, tv in task_vectors.items():
        norm_task_vectors[task_name] = {}
        for k, v in tv.items():
            norm_task_vectors[task_name][k] = v / mean_abs_mags[task_name]
            
    # Seeds to evaluate
    seeds = [42, 100, 2026, 777, 999]
    keep_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    # We sweep alpha over range [0.0001 to 0.0030] since the normalized task vectors are scaled up by ~1/0.001 = 1000x.
    # Standard alpha is ~0.5 to 1.0. With normalized vectors, the effective alpha should be standard_alpha * mean_abs_mag.
    # Since mean_abs_mag is around 0.0010 to 0.0017, the normalized alpha should sweep around 0.0001 to 0.0030.
    alphas_norm = [0.0001, 0.0003, 0.0005, 0.0007, 0.0010, 0.0012, 0.0015, 0.0018, 0.0022, 0.0026, 0.0030]
    
    results = {"GQ-Norm": [], "LQ-Norm": []}
    
    for s_idx, seed in enumerate(seeds):
        print(f"\n=================== RUNNING SEED {seed} ({s_idx+1}/{len(seeds)}) ===================")
        train_loaders, test_loaders, val_data = prepare_dataloaders(datasets, seed)
        
        for masking_type in ["GQ", "LQ"]:
            best_sg_ta_val = -1
            best_k = 0.5
            best_alpha = 0.001
            
            # 1. Sweep on validation data to calibrate parameters
            for k in keep_ratios:
                masked_tvs = apply_sg_ta_masking(norm_task_vectors, base_state, k, masking_type)
                for alpha in alphas_norm:
                    weights = {task: alpha for task in task_names}
                    backbone = construct_merged_backbone(base_state, masked_tvs, weights)
                    val_accs = evaluate_merged_model(backbone, expert_heads, test_loaders, val_data)
                    if val_accs["Joint Mean"] > best_sg_ta_val:
                        best_sg_ta_val = val_accs["Joint Mean"]
                        best_k = k
                        best_alpha = alpha
                        
            # 2. Evaluate optimal on test data
            opt_masked_tvs = apply_sg_ta_masking(norm_task_vectors, base_state, best_k, masking_type)
            opt_weights = {task: best_alpha for task in task_names}
            opt_backbone = construct_merged_backbone(base_state, opt_masked_tvs, opt_weights)
            test_accs = evaluate_merged_model(opt_backbone, expert_heads, test_loaders, None)
            
            method_key = f"{masking_type}-Norm"
            results[method_key].append({
                "k": best_k,
                "alpha": best_alpha,
                "val_acc": best_sg_ta_val,
                "test_accs": test_accs
            })
            print(f"Seed {seed} SG-TA ({method_key}) (k={best_k:.2f}, alpha={best_alpha:.4f}) Test Acc: {test_accs['Joint Mean']:.2f}%")
            print(f"Details: MNIST={test_accs['MNIST']:.2f}%, Fashion={test_accs['FashionMNIST']:.2f}%, CIFAR10={test_accs['CIFAR10']:.2f}%, SVHN={test_accs['SVHN']:.2f}%")

    print("\n" + "="*30 + " AGGREGATE RESULTS FOR NORMALIZED SG-TA " + "="*30)
    for m in ["GQ-Norm", "LQ-Norm"]:
        joint_accs = [res["test_accs"]["Joint Mean"] for res in results[m]]
        mean_joint = np.mean(joint_accs)
        std_joint = np.std(joint_accs)
        
        # Compute mean per dataset
        mean_ds = {}
        for d in task_names:
            mean_ds[d] = np.mean([res["test_accs"][d] for res in results[m]])
            
        print(f"SG-TA ({m}) Joint Mean Accuracy: {mean_joint:.2f}% ± {std_joint:.2f}%")
        print(f"Per-dataset: MNIST={mean_ds['MNIST']:.2f}%, FashionMNIST={mean_ds['FashionMNIST']:.2f}%, CIFAR10={mean_ds['CIFAR10']:.2f}%, SVHN={mean_ds['SVHN']:.2f}%")
        
    # Save results to a file for easy reading
    with open("./results/norm_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
