import os
import json
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# Import functions from train_and_calibrate
from train_and_calibrate import (
    set_seed,
    get_deterministic_subset,
    get_balanced_subset,
    perform_bnc,
    perform_sp_taac,
    perform_hybrid,
    evaluate_model
)

def evaluate_sizes(seed=42):
    print(f"\n================ Evaluating Calibration Sizes for Seed {seed} ================")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Datasets Preparation
    transform_grayscale = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading datasets...")
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_grayscale)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_grayscale)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_grayscale)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_grayscale)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    
    # Subsample test sets for 5x speedup (2,000 samples per task is highly representative)
    mnist_test_sub = get_deterministic_subset(mnist_test, 2000, seed)
    fmnist_test_sub = get_deterministic_subset(fmnist_test, 2000, seed)
    cifar_test_sub = get_deterministic_subset(cifar_test, 2000, seed)
    
    test_loaders = [
        DataLoader(mnist_test_sub, batch_size=128, shuffle=False),
        DataLoader(fmnist_test_sub, batch_size=128, shuffle=False),
        DataLoader(cifar_test_sub, batch_size=128, shuffle=False)
    ]
    
    tasks = ["MNIST", "FashionMNIST", "CIFAR10"]
    scenarios = {
        "A": {"name": "SGD_LowReg", "opt": "sgd", "lr": 1e-4, "wd": 1e-4},
        "B": {"name": "SGD_HighReg", "opt": "sgd", "lr": 1e-4, "wd": 1e-2},
        "C": {"name": "AdamW_LowReg", "opt": "adamw", "lr": 1e-4, "wd": 1e-4},
        "D": {"name": "AdamW_HighLR", "opt": "adamw", "lr": 1e-3, "wd": 1e-4},
        "E": {"name": "AdamW_HighReg", "opt": "adamw", "lr": 1e-4, "wd": 1e-2}
    }
    
    cal_sizes = [16, 32, 64, 128]
    results = {size: {} for size in cal_sizes}
    
    for sc_id, sc in scenarios.items():
        print(f"\n--- Scenario {sc_id}: {sc['name']} ---")
        
        # Load expert models (must exist on disk)
        expert_models = []
        for task_name in tasks:
            model_path = f"models/{sc['name']}/{task_name}_seed{seed}.pt"
            model = resnet18()
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            expert_models.append(model)
            
        # Get weight averaging state (merged weights) on CPU cloned
        expert_states = [m.state_dict() for m in expert_models]
        merged_state = {}
        # We use one of the expert state dicts to get the keys
        for key in expert_states[0].keys():
            merged_state[key] = sum(states[key].cpu().clone() for states in expert_states) / 3.0
            
        merged_model_WA = resnet18()
        merged_model_WA.fc = nn.Linear(merged_model_WA.fc.in_features, 10)
        merged_model_WA = merged_model_WA.to(device)
        
        for size in cal_sizes:
            print(f"  Evaluating calibration size N = {size}")
            # Get size-sample class-balanced calibration subsets
            mnist_cal = get_balanced_subset(mnist_train_full, size, seed)
            fmnist_cal = get_balanced_subset(fmnist_train_full, size, seed)
            cifar_cal = get_balanced_subset(cifar_train_full, size, seed)
            
            cal_loaders = [
                DataLoader(mnist_cal, batch_size=size, shuffle=False),
                DataLoader(fmnist_cal, batch_size=size, shuffle=False),
                DataLoader(cifar_cal, batch_size=size, shuffle=False)
            ]
            
            # Helper function to get a fresh uncalibrated merged model
            def get_fresh_merged():
                m = resnet18()
                m.fc = nn.Linear(m.fc.in_features, 10)
                m.load_state_dict({k: v.clone() for k, v in merged_state.items()})
                return m.to(device)
            
            # 1. Uncalibrated (doesn't depend on calibration size, but good baseline check)
            model_eval = get_fresh_merged()
            accs_uncal = evaluate_model(model_eval, test_loaders, device)
            avg_uncal = sum(accs_uncal) / len(accs_uncal)
            
            # 2. BNC Calibration
            model_eval = get_fresh_merged()
            perform_bnc(model_eval, cal_loaders, device)
            accs_bnc = evaluate_model(model_eval, test_loaders, device)
            avg_bnc = sum(accs_bnc) / len(accs_bnc)
            
            # 3. SP-TAAC Calibration
            model_eval = get_fresh_merged()
            perform_sp_taac(model_eval, expert_models, cal_loaders, device)
            accs_sptaac = evaluate_model(model_eval, test_loaders, device)
            avg_sptaac = sum(accs_sptaac) / len(accs_sptaac)
            
            # 4. Hybrid (Rank 4)
            model_eval = get_fresh_merged()
            perform_hybrid(model_eval, expert_models, cal_loaders, device, rank=4, reg_strength=0.1)
            accs_hybrid4 = evaluate_model(model_eval, test_loaders, device)
            avg_hybrid4 = sum(accs_hybrid4) / len(accs_hybrid4)
            
            # 5. Hybrid (Rank 8)
            model_eval = get_fresh_merged()
            perform_hybrid(model_eval, expert_models, cal_loaders, device, rank=8, reg_strength=0.1)
            accs_hybrid8 = evaluate_model(model_eval, test_loaders, device)
            avg_hybrid8 = sum(accs_hybrid8) / len(accs_hybrid8)
            
            results[size][sc_id] = {
                "uncal_avg": avg_uncal,
                "bnc_avg": avg_bnc,
                "sptaac_avg": avg_sptaac,
                "hybrid4_avg": avg_hybrid4,
                "hybrid8_avg": avg_hybrid8
            }
            print(f"    N={size}: Uncal: {avg_uncal:.2f}%, BNC: {avg_bnc:.2f}%, SP-TAAC: {avg_sptaac:.2f}%, Hybrid4: {avg_hybrid4:.2f}%, Hybrid8: {avg_hybrid8:.2f}%")
            
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    all_results = evaluate_sizes(seed=args.seed)
    
    # Save results to a json file
    with open(f"results_cal_sizes_seed{args.seed}.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved size results for seed {args.seed} successfully!")
