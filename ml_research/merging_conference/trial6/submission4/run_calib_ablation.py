import os
import json
import torch
from torch.utils.data import Subset
import numpy as np
from run_experiments import (
    set_seed,
    get_modified_resnet18,
    get_datasets,
    compute_layer_wise_fisher,
    compute_prototypes,
    build_test_streams,
    run_evaluation
)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load datasets
    (mnist_train, mnist_test), (fmnist_train, fmnist_test), (kmnist_train, kmnist_test) = get_datasets()
    
    # 2. Load experts
    experts = []
    expert_paths = ["expert_mnist.pt", "expert_fmnist.pt", "expert_kmnist.pt"]
    expert_names = ["MNIST", "FashionMNIST", "KMNIST"]
    expert_datasets = [mnist_train, fmnist_train, kmnist_train]
    
    for path, name in zip(expert_paths, expert_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model path {path} not found! Please run run_experiments.py first.")
        model = get_modified_resnet18(num_classes=10).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        experts.append(model)
        
    # Base model (modified ResNet-18 with default weights)
    base_model = get_modified_resnet18(num_classes=10).to(device)
    
    # Build streams (we use the sequential stream for ablation evaluation)
    _, seq_batches = build_test_streams(mnist_test, fmnist_test, kmnist_test)
    
    calib_sizes = [50, 100, 200, 500, 1000]
    results = {}
    
    for size in calib_sizes:
        print(f"\nEvaluating Calibration Size: N_calib = {size}")
        
        # Compute Fisher and Prototypes with current N_calib
        group_fishers = []
        prototypes = []
        for k in range(3):
            fish = compute_layer_wise_fisher(experts[k], expert_datasets[k], device, num_samples=size)
            group_fishers.append(fish)
            
            protos = compute_prototypes(experts[k], expert_datasets[k], device, num_samples=size)
            prototypes.append(protos)
            
        # Evaluate on Sequential Stream under Gaussian Noise and Contrast Shift
        acc_gn, _, _ = run_evaluation(
            base_model, experts, seq_batches, prototypes, group_fishers, "gaussian_noise", "ours", device
        )
        acc_cs, _, _ = run_evaluation(
            base_model, experts, seq_batches, prototypes, group_fishers, "contrast_shift", "ours", device
        )
        
        print(f"  N_calib: {size:<4} | Gaussian Noise Acc: {acc_gn:.2f}% | Contrast Shift Acc: {acc_cs:.2f}%")
        results[str(size)] = {
            "gaussian_noise": acc_gn,
            "contrast_shift": acc_cs
        }
        
    # Save results to JSON
    with open("calib_ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nCalibration ablation study complete! Saved results to calib_ablation_results.json")

if __name__ == "__main__":
    main()
