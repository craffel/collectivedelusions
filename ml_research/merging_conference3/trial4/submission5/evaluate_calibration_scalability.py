import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt

from train_and_merge import (
    set_seed, load_task_datasets, prepare_dataloaders,
    get_task_vector, evaluate_merged_model, construct_merged_backbone
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define task-specific masking helper
def apply_sg_ta_masking_task_specific(task_vectors, base_state, keep_ratios_dict, masking_type="GQ"):
    masked_task_vectors = {}
    for task_name, tv in task_vectors.items():
        masked_task_vectors[task_name] = {}
        k_val = keep_ratios_dict[task_name]
        
        if masking_type == "GQ":
            # Global Quantile (GQ) Masking
            all_vals = []
            for k, v in tv.items():
                all_vals.append(v.flatten())
            all_vals_tensor = torch.cat(all_vals)
            num_keep = int(k_val * len(all_vals_tensor))
            if num_keep == 0:
                num_keep = 1
            threshold = torch.topk(torch.abs(all_vals_tensor), num_keep).values[-1]
            
            for k, v in tv.items():
                mask = torch.abs(v) >= threshold
                masked_task_vectors[task_name][k] = torch.where(mask, v, torch.zeros_like(v))
                
        elif masking_type == "LQ":
            # Layer-wise Quantile (LQ) Masking
            for k, v in tv.items():
                flat_v = v.flatten()
                num_keep = int(k_val * len(flat_v))
                if num_keep == 0:
                    num_keep = 1
                threshold = torch.topk(torch.abs(flat_v), num_keep).values[-1]
                mask = torch.abs(v) >= threshold
                masked_task_vectors[task_name][k] = torch.where(mask, v, torch.zeros_like(v))
                
    return masked_task_vectors

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
    keep_ratios_choices = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    alphas_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Structure for holding results across seeds
    results = {
        "Uniform Grid Search": [],
        "Non-Uniform Random Search": [],
        "Non-Uniform Coordinate Search": []
    }
    
    for s_idx, seed in enumerate(seeds):
        print(f"\n=================== RUNNING CALIBRATION SEED {seed} ({s_idx+1}/{len(seeds)}) ===================")
        train_loaders, test_loaders, val_data = prepare_dataloaders(datasets, seed)
        
        # 1. Uniform Grid Search
        # Budget: 60 evaluations
        print("\n--- Running Uniform Grid Search ---")
        start_time = time.time()
        best_uniform_val = -1
        best_k_uni = 0.5
        best_alpha_uni = 0.3
        uni_eval_count = 0
        
        for k in keep_ratios_choices:
            # Mask task vectors with uniform keep-ratio
            keep_ratios_dict = {task: k for task in task_names}
            masked_tvs = apply_sg_ta_masking_task_specific(task_vectors, base_state, keep_ratios_dict, "GQ")
            for alpha in alphas_choices:
                weights = {task: alpha for task in task_names}
                backbone = construct_merged_backbone(base_state, masked_tvs, weights)
                uni_eval_count += 1
                val_accs = evaluate_merged_model(backbone, expert_heads, test_loaders, val_data)
                if val_accs["Joint Mean"] > best_uniform_val:
                    best_uniform_val = val_accs["Joint Mean"]
                    best_k_uni = k
                    best_alpha_uni = alpha
                    
        uni_time = time.time() - start_time
        
        # Evaluate Uniform Grid Search on test data
        opt_k_dict = {task: best_k_uni for task in task_names}
        opt_masked_tvs = apply_sg_ta_masking_task_specific(task_vectors, base_state, opt_k_dict, "GQ")
        opt_weights = {task: best_alpha_uni for task in task_names}
        opt_backbone = construct_merged_backbone(base_state, opt_masked_tvs, opt_weights)
        test_accs_uni = evaluate_merged_model(opt_backbone, expert_heads, test_loaders, None)
        
        results["Uniform Grid Search"].append({
            "seed": seed,
            "best_k": {task: best_k_uni for task in task_names},
            "best_alpha": {task: best_alpha_uni for task in task_names},
            "val_acc": best_uniform_val,
            "test_accs": test_accs_uni,
            "eval_count": uni_eval_count,
            "time_seconds": uni_time
        })
        print(f"Uniform GS Done. Time: {uni_time:.2f}s, Eval Count: {uni_eval_count}, Test Acc: {test_accs_uni['Joint Mean']:.2f}% (k={best_k_uni:.2f}, alpha={best_alpha_uni:.2f})")
        
        # 2. Non-Uniform Random Search
        # Budget: 60 evaluations (same as Grid Search)
        print("\n--- Running Non-Uniform Random Search ---")
        start_time = time.time()
        best_random_val = -1
        best_k_rand = {}
        best_alpha_rand = {}
        rand_eval_count = 0
        
        # Ensure first evaluation is the best found uniform grid search so we at least start there
        # For fairness, we'll draw 60 completely random combinations of task-specific parameters
        random.seed(seed)
        np.random.seed(seed)
        
        for _ in range(60):
            # Sample random task-specific keep ratios and scaling factors
            k_dict = {task: random.choice(keep_ratios_choices) for task in task_names}
            alpha_dict = {task: random.choice(alphas_choices) for task in task_names}
            
            masked_tvs = apply_sg_ta_masking_task_specific(task_vectors, base_state, k_dict, "GQ")
            backbone = construct_merged_backbone(base_state, masked_tvs, alpha_dict)
            rand_eval_count += 1
            val_accs = evaluate_merged_model(backbone, expert_heads, test_loaders, val_data)
            
            if val_accs["Joint Mean"] > best_random_val:
                best_random_val = val_accs["Joint Mean"]
                best_k_rand = k_dict
                best_alpha_rand = alpha_dict
                
        rand_time = time.time() - start_time
        
        # Evaluate Non-Uniform Random Search on test data
        opt_masked_tvs_rand = apply_sg_ta_masking_task_specific(task_vectors, base_state, best_k_rand, "GQ")
        opt_backbone_rand = construct_merged_backbone(base_state, opt_masked_tvs_rand, best_alpha_rand)
        test_accs_rand = evaluate_merged_model(opt_backbone_rand, expert_heads, test_loaders, None)
        
        results["Non-Uniform Random Search"].append({
            "seed": seed,
            "best_k": best_k_rand,
            "best_alpha": best_alpha_rand,
            "val_acc": best_random_val,
            "test_accs": test_accs_rand,
            "eval_count": rand_eval_count,
            "time_seconds": rand_time
        })
        print(f"Non-Uniform RS Done. Time: {rand_time:.2f}s, Eval Count: {rand_eval_count}, Test Acc: {test_accs_rand['Joint Mean']:.2f}%")
        print(f"  Chosen k: {best_k_rand}")
        print(f"  Chosen alpha: {best_alpha_rand}")
        
        # 3. Non-Uniform Coordinate Search
        # Budget: 4 tasks * 11 evaluations = 44 evaluations
        print("\n--- Running Non-Uniform Coordinate Search ---")
        start_time = time.time()
        
        # Initialize with reasonable default uniform values (e.g. k=0.5, alpha=0.5)
        current_k = {task: 0.5 for task in task_names}
        current_alpha = {task: 0.5 for task in task_names}
        coord_eval_count = 0
        best_coord_val = -1
        
        # We perform coordinate descent sequentially across tasks
        # To avoid dependency on order, we can run 1 pass of coordinate search
        for task in task_names:
            best_task_val = -1
            best_task_k = current_k[task]
            best_task_alpha = current_alpha[task]
            
            # Sweep k and alpha for this specific task, holding others fixed
            # To keep budget reasonable, we sweep over a coarser grid of 5 k values and 5 alpha values
            # Coarser task-specific grid: k in [0.1, 0.3, 0.5, 0.7, 1.0], alpha in [0.1, 0.3, 0.5, 0.7, 1.0] (25 evaluations per task, 100 total)
            k_sweep = [0.1, 0.3, 0.5, 0.7, 1.0]
            alpha_sweep = [0.1, 0.3, 0.5, 0.7, 1.0]
            
            for k_val in k_sweep:
                # Set temporary k for this task
                temp_k = current_k.copy()
                temp_k[task] = k_val
                
                # Apply masking
                masked_tvs = apply_sg_ta_masking_task_specific(task_vectors, base_state, temp_k, "GQ")
                
                for alpha_val in alpha_sweep:
                    # Set temporary alpha for this task
                    temp_alpha = current_alpha.copy()
                    temp_alpha[task] = alpha_val
                    
                    backbone = construct_merged_backbone(base_state, masked_tvs, temp_alpha)
                    coord_eval_count += 1
                    val_accs = evaluate_merged_model(backbone, expert_heads, test_loaders, val_data)
                    
                    if val_accs["Joint Mean"] > best_task_val:
                        best_task_val = val_accs["Joint Mean"]
                        best_task_k = k_val
                        best_task_alpha = alpha_val
                        
            # Update coordinate parameters for this task
            current_k[task] = best_task_k
            current_alpha[task] = best_task_alpha
            best_coord_val = best_task_val
            print(f"  Optimized coordinate for {task}: k={best_task_k:.2f}, alpha={best_task_alpha:.2f}. Val Acc: {best_task_val:.2f}%")
            
        coord_time = time.time() - start_time
        
        # Evaluate Non-Uniform Coordinate Search on test data
        opt_masked_tvs_coord = apply_sg_ta_masking_task_specific(task_vectors, base_state, current_k, "GQ")
        opt_backbone_coord = construct_merged_backbone(base_state, opt_masked_tvs_coord, current_alpha)
        test_accs_coord = evaluate_merged_model(opt_backbone_coord, expert_heads, test_loaders, None)
        
        results["Non-Uniform Coordinate Search"].append({
            "seed": seed,
            "best_k": current_k.copy(),
            "best_alpha": current_alpha.copy(),
            "val_acc": best_coord_val,
            "test_accs": test_accs_coord,
            "eval_count": coord_eval_count,
            "time_seconds": coord_time
        })
        print(f"Non-Uniform Coordinate Search Done. Time: {coord_time:.2f}s, Eval Count: {coord_eval_count}, Test Acc: {test_accs_coord['Joint Mean']:.2f}%")
        print(f"  Chosen k: {current_k}")
        print(f"  Chosen alpha: {current_alpha}")

    # ----------------- Print and Save Summary -----------------
    print("\n" + "="*30 + " SUMMARY OF CALIBRATION SCALABILITY ANALYSIS " + "="*30)
    for m in results.keys():
        joint_accs = [res["test_accs"]["Joint Mean"] for res in results[m]]
        mean_joint = np.mean(joint_accs)
        std_joint = np.std(joint_accs)
        mean_evals = np.mean([res["eval_count"] for res in results[m]])
        mean_time = np.mean([res["time_seconds"] for res in results[m]])
        
        # Compute mean per dataset
        mean_ds = {}
        for d in task_names:
            mean_ds[d] = np.mean([res["test_accs"][d] for res in results[m]])
            
        print(f"\nMethod: {m}")
        print(f"  Joint Mean Test Accuracy: {mean_joint:.2f}% ± {std_joint:.2f}%")
        print(f"  Per-Dataset Accuracy: MNIST={mean_ds['MNIST']:.2f}%, FashionMNIST={mean_ds['FashionMNIST']:.2f}%, CIFAR10={mean_ds['CIFAR10']:.2f}%, SVHN={mean_ds['SVHN']:.2f}%")
        print(f"  Calibration Budget: {mean_evals:.1f} evaluations")
        print(f"  Calibration Time: {mean_time:.2f} seconds")
        
    with open("./results/calibration_scalability_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved scalability metrics to ./results/calibration_scalability_metrics.json")

if __name__ == "__main__":
    main()
