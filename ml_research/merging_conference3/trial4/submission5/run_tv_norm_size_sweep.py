import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import timm
from train_and_merge import (
    set_seed, load_task_datasets, prepare_dataloaders, get_task_vector,
    evaluate_merged_model, apply_sg_ta_masking, construct_merged_backbone
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dynamic validation loader helper for larger split sizes
def prepare_val_data_for_size(datasets, size, seed):
    val_data = {}
    for task_name, task_ds in datasets.items():
        indices = list(range(len(task_ds["train"])))
        random.seed(seed)
        val_indices = random.sample(indices, size)
        
        val_images = []
        val_labels = []
        for idx in val_indices:
            img, label = task_ds["train"][idx]
            val_images.append(img)
            val_labels.append(label)
            
        val_data[task_name] = {
            "images": torch.stack(val_images).to(device),
            "labels": torch.tensor(val_labels).to(device)
        }
    return val_data

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
    
    # Compute mean absolute magnitudes for TV-Norm
    mean_abs_mags = {}
    for task_name, tv in task_vectors.items():
        total_abs_sum = 0.0
        total_numel = 0
        for k, v in tv.items():
            total_abs_sum += torch.sum(torch.abs(v)).item()
            total_numel += v.numel()
        mean_abs_mags[task_name] = total_abs_sum / total_numel
        
    # Create normalized task vectors
    norm_task_vectors = {}
    for task_name, tv in task_vectors.items():
        norm_task_vectors[task_name] = {}
        for k, v in tv.items():
            norm_task_vectors[task_name][k] = v / mean_abs_mags[task_name]
            
    seeds = [42, 100, 2026, 777, 999]
    keep_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    alphas_norm = [0.0001, 0.0003, 0.0005, 0.0007, 0.0010, 0.0012, 0.0015, 0.0018, 0.0022, 0.0026, 0.0030]
    
    val_sizes = [10, 20, 50, 100]
    results = {}
    
    for size in val_sizes:
        print(f"\n--- Running TV-Norm sweep for validation size N_val = {size} ---")
        size_results = []
        
        for s_idx, seed in enumerate(seeds):
            # Prepare loaders and validation splits of specific size
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Prepare loaders and validation split
            # We can re-use train_loaders and test_loaders since they are standard
            # but we construct a customized val_data of size 'size'
            val_data = prepare_val_data_for_size(datasets, size, seed)
            _, test_loaders, _ = prepare_dataloaders(datasets, seed)
            
            best_val_acc = -1
            best_k = 0.5
            best_alpha = 0.001
            
            # 1. Sweep on validation subset to calibrate parameters
            for k in keep_ratios:
                masked_tvs = apply_sg_ta_masking(norm_task_vectors, base_state, k, "GQ")
                for alpha in alphas_norm:
                    weights = {task: alpha for task in task_names}
                    backbone = construct_merged_backbone(base_state, masked_tvs, weights)
                    val_accs = evaluate_merged_model(backbone, expert_heads, test_loaders, val_data)
                    if val_accs["Joint Mean"] > best_val_acc:
                        best_val_acc = val_accs["Joint Mean"]
                        best_k = k
                        best_alpha = alpha
                        
            # 2. Evaluate optimal on test sets
            opt_masked_tvs = apply_sg_ta_masking(norm_task_vectors, base_state, best_k, "GQ")
            opt_weights = {task: best_alpha for task in task_names}
            opt_backbone = construct_merged_backbone(base_state, opt_masked_tvs, opt_weights)
            test_accs = evaluate_merged_model(opt_backbone, expert_heads, test_loaders, None)
            
            size_results.append(test_accs["Joint Mean"])
            print(f"Seed {seed}: optimal k={best_k:.2f}, alpha={best_alpha:.4f}, test Joint Mean Acc: {test_accs['Joint Mean']:.2f}%")
            
        mean_joint = np.mean(size_results)
        std_joint = np.std(size_results)
        results[size] = {
            "mean": mean_joint,
            "std": std_joint,
            "raw": size_results
        }
        print(f"N_val = {size} Summary: Joint Mean Accuracy = {mean_joint:.2f}% ± {std_joint:.2f}%")
        
    with open("./results/tv_norm_size_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved TV-Norm size-sweep metrics to ./results/tv_norm_size_metrics.json")

if __name__ == "__main__":
    main()
