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
    run_tta_evaluation,
    get_pretrained_base_encoder,
    get_task_vector,
    ResNet18Expert,
    compute_diagonal_fisher,
    evaluate_static_merged
)

def generate_chunked_test_stream(test_datasets, chunk_size, num_batches_per_task=50, batch_size=32, seed=42):
    # Set seed for reproducibility of stream generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    loaders = {
        0: DataLoader(test_datasets[0], batch_size=batch_size, shuffle=True),
        1: DataLoader(test_datasets[1], batch_size=batch_size, shuffle=True),
        2: DataLoader(test_datasets[2], batch_size=batch_size, shuffle=True)
    }
    
    iters = [iter(loaders[i]) for i in range(3)]
    
    batches = []
    task_counts = {0: 0, 1: 0, 2: 0}
    active_task = 0
    
    while sum(task_counts.values()) < 3 * num_batches_per_task:
        remaining_for_task = num_batches_per_task - task_counts[active_task]
        current_chunk = min(chunk_size, remaining_for_task)
        
        for _ in range(current_chunk):
            try:
                x, y = next(iters[active_task])
            except StopIteration:
                iters[active_task] = iter(loaders[active_task])
                x, y = next(iters[active_task])
            batches.append((active_task, x, y))
            task_counts[active_task] += 1
            
        for offset in range(1, 4):
            candidate = (active_task + offset) % 3
            if task_counts[candidate] < num_batches_per_task:
                active_task = candidate
                break
                
    return batches

def main():
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

    # Base encoder and task vectors
    base_encoder = get_pretrained_base_encoder()
    base_encoder.eval()
    task_vectors = [get_task_vector(expert_encoders[i], base_encoder) for i in range(3)]

    # Freeze encoders
    base_encoder.requires_grad_(False)
    for enc in expert_encoders:
        enc.requires_grad_(False)
        enc.eval()

    # Pre-compute diagonal Fisher for expert heads (N=200 Clean baseline setting)
    print("\n--- Pre-computing Fisher Information for EWC-TTA ---")
    fisher_diags = []
    for i, task in enumerate(tasks):
        train_set, _ = data[task]
        subset_indices = list(range(200))
        subset = Subset(train_set, subset_indices)
        fim = compute_diagonal_fisher(expert_encoders[i], expert_heads[i], subset, num_samples=200)
        fisher_diags.append(fim)

    test_datasets = [data[tasks[i]][1] for i in range(3)]
    
    chunk_sizes = [1, 2, 5, 10, 25, 50]
    
    lr_lambda = 0.50
    gamma = 100.0
    
    results = {}
    
    print("\n--- Evaluating Across Task Switching Frequencies (Chunk Size M) ---")
    
    for M in chunk_sizes:
        print(f"\n>> Chunk Size M = {M} batches")
        batches = generate_chunked_test_stream(test_datasets, M, seed=42)
        
        # 1. Static Merged
        static_acc = evaluate_static_merged(base_encoder, task_vectors, expert_heads, batches)
        print(f"Static Merged Acc: {static_acc:.2f}%")
        
        # 2. Standard TTA (Gamma = 0)
        std_acc = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=batches,
            reg_type='none',
            gamma=0.0,
            tta_lr_lambdas=lr_lambda
        )
        print(f"Standard TTA Acc: {std_acc:.2f}%")
        
        # 3. L2 TTA (Gamma = 100.0)
        l2_acc = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=batches,
            reg_type='l2',
            gamma=gamma,
            tta_lr_lambdas=lr_lambda
        )
        print(f"L2-TTA (G=100) Acc: {l2_acc:.2f}%")
        
        # 4. EWC TTA (Ours, Gamma = 100.0)
        ewc_acc = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=batches,
            fisher_diags=fisher_diags,
            reg_type='ewc',
            gamma=gamma,
            tta_lr_lambdas=lr_lambda
        )
        print(f"EWC-TTA (G=100) Acc: {ewc_acc:.2f}%")
        
        results[M] = {
            'static': static_acc,
            'standard': std_acc,
            'l2': l2_acc,
            'ewc': ewc_acc
        }

    # Print results summary table
    print("\n" + "="*80)
    print(f"TASK SWITCHING FREQUENCY STUDY (M = Chunk Size, LR_LAMBDA={lr_lambda}, GAMMA={gamma})")
    print("="*80)
    print(f"{'Chunk Size M':12} | {'Static Merged':14} | {'Standard TTA':14} | {'L2-TTA (G=100)':14} | {'EWC-TTA (Ours)':14}")
    print("-"*80)
    for M in chunk_sizes:
        res = results[M]
        print(f"{M:12d} | {res['static']:13.2f}% | {res['standard']:13.2f}% | {res['l2']:13.2f}% | {res['ewc']:13.2f}%")
    print("="*80)

    # Save summary results to a markdown table
    with open("switching_frequency_results.txt", "w") as f:
        f.write("# Task Switching Frequency Sweep Results\n\n")
        f.write(f"Parameters: $\\eta_\\lambda = {lr_lambda}$, $\\gamma = {gamma}$\n\n")
        f.write("| Chunk Size M | Static Merged | Standard TTA | L2-TTA (\\gamma=100) | EWC-TTA (Ours) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for M in chunk_sizes:
            res = results[M]
            f.write(f"| {M} | {res['static']:.2f}% | {res['standard']:.2f}% | {res['l2']:.2f}% | {res['ewc']:.2f}% |\n")

if __name__ == "__main__":
    main()
