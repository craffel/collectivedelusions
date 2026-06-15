import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
    # Use a small validation loader or test loader for quick evaluation
    train_loaders, test_loaders, val_data = prepare_dataloaders(datasets, 42)
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
        
    print("Loaded expert states and extracted task vectors.")
    
    # Print mean absolute magnitudes
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
            
    print("\nEvaluating Standard SG-TA (GQ) at k=0.3, alpha=0.8...")
    masked_tvs = apply_sg_ta_masking(task_vectors, base_state, 0.3, "GQ")
    weights = {task: 0.8 for task in task_names}
    backbone = construct_merged_backbone(base_state, masked_tvs, weights)
    # Evaluate on a subset of test loaders to make it extremely fast on CPU
    fast_test_loaders = {}
    for task, loader in test_loaders.items():
        # Let's take only first 1 batch
        fast_test_loaders[task] = [next(iter(loader))]
        
    test_accs = evaluate_merged_model(backbone, expert_heads, fast_test_loaders, None)
    print(f"Standard Test Accs (on 1 batch): {test_accs}")
    
    print("\nEvaluating Normalized SG-TA (GQ) at k=0.3, alpha=0.001...")
    norm_masked_tvs = apply_sg_ta_masking(norm_task_vectors, base_state, 0.3, "GQ")
    norm_weights = {task: 0.001 for task in task_names}
    norm_backbone = construct_merged_backbone(base_state, norm_masked_tvs, norm_weights)
    test_norm_accs = evaluate_merged_model(norm_backbone, expert_heads, fast_test_loaders, None)
    print(f"Normalized Test Accs (on 1 batch): {test_norm_accs}")

if __name__ == "__main__":
    main()
