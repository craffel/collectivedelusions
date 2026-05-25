import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
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
    compute_diagonal_fisher
)

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

    # Base encoder and task vectors
    base_encoder = get_pretrained_base_encoder()
    base_encoder.eval()
    task_vectors = [get_task_vector(expert_encoders[i], base_encoder) for i in range(3)]

    # Freeze encoders
    base_encoder.requires_grad_(False)
    for enc in expert_encoders:
        enc.requires_grad_(False)
        enc.eval()

    # Define Fisher scenarios
    # 1. N = 10 Clean
    # 2. N = 50 Clean
    # 3. N = 200 Clean (Baseline)
    # 4. N = 200 Synthetic (Gaussian Noise)
    # 5. N = 200 Out-of-Distribution (OOD)
    
    scenarios = ['N=10 Clean', 'N=50 Clean', 'N=200 Clean', 'N=200 Synthetic', 'N=200 OOD']
    fisher_by_scenario = {}

    print("\n--- Computing Fisher Information for Scenarios ---")
    
    for scenario in scenarios:
        print(f"\n>> Scenario: {scenario}")
        fisher_diags = []
        for i, task in enumerate(tasks):
            train_set, _ = data[task]
            
            if scenario == 'N=10 Clean':
                subset_indices = list(range(10))
                subset = Subset(train_set, subset_indices)
                fim = compute_diagonal_fisher(expert_encoders[i], expert_heads[i], subset, num_samples=10)
            elif scenario == 'N=50 Clean':
                subset_indices = list(range(50))
                subset = Subset(train_set, subset_indices)
                fim = compute_diagonal_fisher(expert_encoders[i], expert_heads[i], subset, num_samples=50)
            elif scenario == 'N=200 Clean':
                subset_indices = list(range(200))
                subset = Subset(train_set, subset_indices)
                fim = compute_diagonal_fisher(expert_encoders[i], expert_heads[i], subset, num_samples=200)
            elif scenario == 'N=200 Synthetic':
                # Generate Gaussian noise of correct shape (3, 224, 224)
                noise_imgs = torch.randn(200, 3, 224, 224)
                # Random labels
                noise_lbls = torch.randint(0, 10, (200,))
                noise_dataset = TensorDataset(noise_imgs, noise_lbls)
                fim = compute_diagonal_fisher(expert_encoders[i], expert_heads[i], noise_dataset, num_samples=200)
            elif scenario == 'N=200 OOD':
                # Use a different task's dataset as OOD
                ood_task = tasks[(i + 1) % 3]
                ood_train_set, _ = data[ood_task]
                subset_indices = list(range(200))
                subset = Subset(ood_train_set, subset_indices)
                fim = compute_diagonal_fisher(expert_encoders[i], expert_heads[i], subset, num_samples=200)
                
            fisher_diags.append(fim)
        fisher_by_scenario[scenario] = fisher_diags

    # Generate test streams once and reuse
    test_datasets = [data[tasks[i]][1] for i in range(3)]
    seq_batches = generate_test_stream(test_datasets, 'sequential', seed=42)
    alt_batches = generate_test_stream(test_datasets, 'alternating', seed=42)

    # We evaluate under eta_lambda = 0.50 and gamma = 100.0 (where EWC is highly regularized)
    # to see how robust EWC is to imperfect or low-data Fisher estimations
    lr_lambda = 0.50
    gamma = 100.0

    results = {}
    
    # Also get Baseline accuracies for comparison
    print("\n--- Running TTA Evaluations ---")
    
    # 1. Standard TTA (Gamma = 0)
    print("Evaluating Standard TTA...")
    std_seq = run_tta_evaluation(
        base_encoder=base_encoder,
        task_vectors=task_vectors,
        expert_encoders=expert_encoders,
        expert_heads=expert_heads,
        batches=seq_batches,
        reg_type='none',
        gamma=0.0,
        tta_lr_lambdas=lr_lambda
    )
    std_alt = run_tta_evaluation(
        base_encoder=base_encoder,
        task_vectors=task_vectors,
        expert_encoders=expert_encoders,
        expert_heads=expert_heads,
        batches=alt_batches,
        reg_type='none',
        gamma=0.0,
        tta_lr_lambdas=lr_lambda
    )
    
    # 2. L2 TTA (Gamma = 100.0)
    print("Evaluating L2 TTA...")
    l2_seq = run_tta_evaluation(
        base_encoder=base_encoder,
        task_vectors=task_vectors,
        expert_encoders=expert_encoders,
        expert_heads=expert_heads,
        batches=seq_batches,
        reg_type='l2',
        gamma=gamma,
        tta_lr_lambdas=lr_lambda
    )
    l2_alt = run_tta_evaluation(
        base_encoder=base_encoder,
        task_vectors=task_vectors,
        expert_encoders=expert_encoders,
        expert_heads=expert_heads,
        batches=alt_batches,
        reg_type='l2',
        gamma=gamma,
        tta_lr_lambdas=lr_lambda
    )

    for scenario in scenarios:
        print(f"Evaluating EWC-TTA with {scenario} Fisher prior...")
        seq_acc = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=seq_batches,
            fisher_diags=fisher_by_scenario[scenario],
            reg_type='ewc',
            gamma=gamma,
            tta_lr_lambdas=lr_lambda
        )
        alt_acc = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=alt_batches,
            fisher_diags=fisher_by_scenario[scenario],
            reg_type='ewc',
            gamma=gamma,
            tta_lr_lambdas=lr_lambda
        )
        results[scenario] = (seq_acc, alt_acc)

    # Print results summary table
    print("\n" + "="*80)
    print(f"FISHER ESTIMATION ABLATION STUDY (LR_LAMBDA={lr_lambda}, GAMMA={gamma})")
    print("="*80)
    print(f"{'Fisher Scenario':25} | {'Sequential Acc':15} | {'Alternating Acc':15}")
    print("-"*80)
    print(f"{'Standard TTA (G=0)':25} | {std_seq:14.2f}% | {std_alt:14.2f}%")
    print(f"{'L2-TTA (G=100)':25} | {l2_seq:14.2f}% | {l2_alt:14.2f}%")
    print("-"*80)
    for scenario in scenarios:
        seq_acc, alt_acc = results[scenario]
        print(f"{'EWC-TTA (' + scenario + ')':25} | {seq_acc:14.2f}% | {alt_acc:14.2f}%")
    print("="*80)

    # Save summary results to a markdown table
    with open("fisher_ablation_results.txt", "w") as f:
        f.write("# Fisher Prior Ablation Study Results\n\n")
        f.write(f"Parameters: $\\eta_\\lambda = {lr_lambda}$, $\\gamma = {gamma}$\n\n")
        f.write("| Fisher Scenario | Sequential Stream Accuracy | Alternating Stream Accuracy |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| Standard TTA ($\\gamma=0$) | {std_seq:.2f}% | {std_alt:.2f}% |\n")
        f.write(f"| L2-TTA ($\\gamma=100$) | {l2_seq:.2f}% | {l2_alt:.2f}% |\n")
        f.write(f"| EWC-TTA (N=10 Clean) | {results['N=10 Clean'][0]:.2f}% | {results['N=10 Clean'][1]:.2f}% |\n")
        f.write(f"| EWC-TTA (N=50 Clean) | {results['N=50 Clean'][0]:.2f}% | {results['N=50 Clean'][1]:.2f}% |\n")
        f.write(f"| EWC-TTA (N=200 Clean) | {results['N=200 Clean'][0]:.2f}% | {results['N=200 Clean'][1]:.2f}% |\n")
        f.write(f"| EWC-TTA (N=200 Synthetic / Noise) | {results['N=200 Synthetic'][0]:.2f}% | {results['N=200 Synthetic'][1]:.2f}% |\n")
        f.write(f"| EWC-TTA (N=200 OOD) | {results['N=200 OOD'][0]:.2f}% | {results['N=200 OOD'][1]:.2f}% |\n")

if __name__ == "__main__":
    main()
