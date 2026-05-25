import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
import copy
import random
import numpy as np

from experiment import (
    get_datasets,
    generate_test_stream,
    run_tta_evaluation,
    get_pretrained_base_encoder,
    get_task_vector,
    ResNet18Expert,
    compute_diagonal_fisher,
    evaluate_static_merged
)

def get_ties_task_vectors(task_vectors, fraction=0.2):
    """
    Applies TIES-Merging steps (Trim, Sign Elect, Resolve Conflict) to individual task vectors.
    """
    ties_task_vectors = [{} for _ in range(len(task_vectors))]
    
    with torch.no_grad():
        keys = task_vectors[0].keys()
        for name in keys:
            # 1. Trim: Keep only top fraction (e.g. 20%) by absolute magnitude
            trimmed_tvs = []
            for tv in task_vectors:
                v = tv[name].clone()
                # If 1D/0D or very small parameter, don't trim
                if v.numel() <= 10:
                    trimmed_tvs.append(v)
                    continue
                
                flat_v = v.flatten()
                k = max(1, int(fraction * flat_v.numel()))
                threshold = torch.topk(flat_v.abs(), k).values[-1]
                mask = flat_v.abs() >= threshold
                flat_v_trimmed = flat_v * mask
                trimmed_tvs.append(flat_v_trimmed.view_as(v))
                
            # 2. Elect Sign
            sign_sum = torch.zeros_like(trimmed_tvs[0])
            for t_v in trimmed_tvs:
                sign_sum += t_v.sign()
            
            majority_sign = sign_sum.sign()
            
            # 3. Resolve Sign Conflict
            for i, t_v in enumerate(trimmed_tvs):
                mask = (t_v.sign() == majority_sign) & (majority_sign != 0)
                ties_task_vectors[i][name] = t_v * mask
                
    return ties_task_vectors

def main():
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    data = get_datasets()
    tasks = ['mnist', 'fmnist', 'kmnist']
    
    # Load expert models
    expert_encoders = []
    expert_heads = []
    for task in tasks:
        resnet = ResNet18Expert()
        resnet.encoder.load_state_dict(torch.load(f"./checkpoints/{task}_encoder.pt", map_location=device))
        resnet.fc.load_state_dict(torch.load(f"./checkpoints/{task}_head.pt", map_location=device))
        expert_encoders.append(resnet.encoder.to(device))
        expert_heads.append(resnet.fc.to(device))

    # Base encoder and standard task vectors
    base_encoder = get_pretrained_base_encoder()
    base_encoder.eval()
    
    # Freeze encoders
    base_encoder.requires_grad_(False)
    for enc in expert_encoders:
        enc.requires_grad_(False)
        enc.eval()

    task_vectors = [get_task_vector(expert_encoders[i], base_encoder) for i in range(3)]

    # Compute TIES-processed task vectors with fraction = 0.2
    ties_task_vectors = get_ties_task_vectors(task_vectors, fraction=0.2)

    # Pre-compute diagonal Fisher for expert heads
    fisher_diags = []
    for i, task in enumerate(tasks):
        train_set, _ = data[task]
        subset_indices = list(range(200))
        subset_train_set = Subset(train_set, subset_indices)
        fim = compute_diagonal_fisher(expert_encoders[i], expert_heads[i], subset_train_set, num_samples=200)
        fisher_diags.append(fim)

    test_datasets = [data[tasks[i]][1] for i in range(3)]
    
    all_results = {}
    
    for stream in ['alternating', 'sequential']:
        print(f"\n--- Running '{stream}' stream with TIES vs. Standard Task Arithmetic ---")
        batches = generate_test_stream(test_datasets, stream_type=stream, num_batches_per_task=50, batch_size=32, seed=42)
        
        # 1. Standard Task Arithmetic - Static
        static_ta = evaluate_static_merged(base_encoder, task_vectors, expert_heads, batches)
        
        # 2. Standard Task Arithmetic - Std TTA
        std_tta_ta = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=batches,
            reg_type='none',
            gamma=0.0,
            tta_lr_lambdas=0.50
        )
        
        # 3. Standard Task Arithmetic - EWC-TTA
        ewc_tta_ta = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=batches,
            fisher_diags=fisher_diags,
            reg_type='ewc',
            gamma=100.0,
            tta_lr_lambdas=0.50
        )
        
        # 4. TIES - Static
        static_ties = evaluate_static_merged(base_encoder, ties_task_vectors, expert_heads, batches)
        
        # 5. TIES - Std TTA
        std_tta_ties = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=ties_task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=batches,
            reg_type='none',
            gamma=0.0,
            tta_lr_lambdas=0.50
        )
        
        # 6. TIES - EWC-TTA
        ewc_tta_ties = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=ties_task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=batches,
            fisher_diags=fisher_diags,
            reg_type='ewc',
            gamma=100.0,
            tta_lr_lambdas=0.50
        )
        
        all_results[stream] = {
            'static_ta': static_ta,
            'std_tta_ta': std_tta_ta,
            'ewc_tta_ta': ewc_tta_ta,
            'static_ties': static_ties,
            'std_tta_ties': std_tta_ties,
            'ewc_tta_ties': ewc_tta_ties
        }

    # Print a summary table of results
    print("\n==========================================================================")
    print("TIES-MERGING VS. TASK ARITHMETIC GNERALIZATION RESULTS (SEED=42, lr_lambda=0.5, G=100)")
    print("==========================================================================")
    print("Method / Stream                | Alternating Stream | Sequential Stream")
    print("--------------------------------------------------------------------------")
    for method_key, method_name in [
        ('static_ta', 'Static Task Arithmetic'),
        ('std_tta_ta', 'Std TTA on Task Arithmetic'),
        ('ewc_tta_ta', 'EWC-TTA on Task Arithmetic'),
        ('static_ties', 'Static TIES-Merging'),
        ('std_tta_ties', 'Std TTA on TIES-Merging'),
        ('ewc_tta_ties', 'EWC-TTA on TIES-Merging')
    ]:
        print(f"{method_name:30} | {all_results['alternating'][method_key]:18.2f}% | {all_results['sequential'][method_key]:17.2f}%")
    print("==========================================================================")
    
    # Save a comparison summary text to a file for easy reading
    with open("ties_results_summary.txt", "w") as f:
        f.write("TIES-MERGING VS. TASK ARITHMETIC GENERALIZATION RESULTS\n")
        f.write("Method / Stream                | Alternating Stream | Sequential Stream\n")
        f.write("--------------------------------------------------------------------------\n")
        for method_key, method_name in [
            ('static_ta', 'Static Task Arithmetic'),
            ('std_tta_ta', 'Std TTA on Task Arithmetic'),
            ('ewc_tta_ta', 'EWC-TTA on Task Arithmetic'),
            ('static_ties', 'Static TIES-Merging'),
            ('std_tta_ties', 'Std TTA on TIES-Merging'),
            ('ewc_tta_ties', 'EWC-TTA on TIES-Merging')
        ]:
            f.write(f"{method_name:30} | {all_results['alternating'][method_key]:18.2f}% | {all_results['sequential'][method_key]:17.2f}%\n")

if __name__ == "__main__":
    main()
