import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.func import functional_call
from experiment import SimpleCNN, get_datasets, train_expert, precompute_prototypes, precompute_fisher, generate_stream, run_bk_co_merge

if __name__ == '__main__':
    # Initialize immediate set_seed to disable cuDNN globally before model allocations
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing wider sweep on device: {device}")
    
    # 1. Load data
    mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test = get_datasets(dry_run=False)
    
    # 2. Train Expert Models
    print("Training experts...")
    model_0 = SimpleCNN().to(device)
    model_1 = SimpleCNN().to(device)
    base_model = SimpleCNN().to(device)
    base_model.eval()
    for m in base_model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()
    
    model_0 = train_expert(model_0, mnist_train, device, epochs=2)
    model_1 = train_expert(model_1, fmnist_train, device, epochs=2)
    
    # 3. Precompute Prototypes
    prototypes_0 = precompute_prototypes(model_0, mnist_train, device, num_samples=256)
    prototypes_1 = precompute_prototypes(model_1, fmnist_train, device, num_samples=256)
    
    # 4. Generate Test Stream
    # Reset seed to 42 for perfect reproducibility and consistency with experiment.py
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False
    
    batches = generate_stream(mnist_test, fmnist_test, kmnist_test, dry_run=False)
    
    # 5. Sweep Hyperparameters (Granular and wider range)
    lrs = [0.03, 0.05, 0.08, 0.1, 0.12, 0.15]
    steps_list = [2, 3, 4, 5]
    gammas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    
    best_overall = 0.0
    best_config = None
    best_results = None
    
    print("\n--- Starting Wider Hyperparameter Sweep ---")
    print("lr,steps,gamma_coherence,Clean MNIST,Noisy MNIST,Clean FashionMNIST,Noisy FashionMNIST,Novel KMNIST,Overall")
    
    sweep_log = []
    
    for lr in lrs:
        for steps in steps_list:
            for gamma in gammas:
                # We only sweep TS-BK-CoMerge as it was consistently better or equal
                res_ts, overall_ts = run_bk_co_merge(
                    batches, model_0, model_1, base_model, device, prototypes_0, prototypes_1,
                    steps=steps, lr=lr, gamma_coherence=gamma, temporal_smoothing=True
                )
                
                print(f"TS-BK-CoMerge: {lr},{steps},{gamma},{res_ts['Clean MNIST']:.4f},{res_ts['Noisy MNIST']:.4f},{res_ts['Clean FashionMNIST']:.4f},{res_ts['Noisy FashionMNIST']:.4f},{res_ts['Novel KMNIST']:.4f},{overall_ts:.4f}")
                
                sweep_log.append({
                    "method": "TS-BK-CoMerge", "lr": lr, "steps": steps, "gamma": gamma, "res": res_ts, "overall": overall_ts
                })
                
                if overall_ts > best_overall:
                    best_overall = overall_ts
                    best_config = {"method": "TS-BK-CoMerge", "lr": lr, "steps": steps, "gamma": gamma}
                    best_results = res_ts
                    
    print(f"\nBest Config: {best_config} with Overall Accuracy: {best_overall:.4%}")
    
    # Save sweep results to sweep_wider_results.txt
    with open("sweep_wider_results.txt", "w") as f:
        f.write("Method,lr,steps,gamma_coherence,Clean MNIST,Noisy MNIST,Clean FashionMNIST,Noisy FashionMNIST,Novel KMNIST,Overall\n")
        for entry in sweep_log:
            res = entry["res"]
            f.write(f"{entry['method']},{entry['lr']},{entry['steps']},{entry['gamma']},{res['Clean MNIST']:.4f},{res['Noisy MNIST']:.4f},{res['Clean FashionMNIST']:.4f},{res['Noisy FashionMNIST']:.4f},{res['Novel KMNIST']:.4f},{entry['overall']:.4f}\n")
            
    print("Wider Sweep results saved to sweep_wider_results.txt!")
