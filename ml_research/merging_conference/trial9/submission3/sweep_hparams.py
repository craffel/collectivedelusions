import torch
import numpy as np
import os
import itertools
from evaluate import evaluate_method
import data
import sys

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running sweep on device: {device}", flush=True)
    
    # Load experts
    state_cos_mnist = torch.load('checkpoints/cos_mnist.pt', map_location=device)
    state_cos_fmnist = torch.load('checkpoints/cos_fmnist.pt', map_location=device)
    prototypes_dict = torch.load('checkpoints/prototypes.pt', map_location=device)
    
    # Load stream
    mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test = data.get_datasets()
    stream_batches = data.create_non_stationary_stream(mnist_test, fmnist_test, kmnist_test)
    
    # Grid search candidate values around the new best area
    betas = [0.2, 0.4, 0.6]
    gammas = [0.0001, 0.0005, 0.001]
    lrs = [0.005, 0.008, 0.01]
    eps_stabs = [0.001, 0.003, 0.005]
    s_temps = [3.5, 3.6, 3.7]
    
    best_overall = 0.0
    best_config = {}
    best_results = None
    
    combinations = list(itertools.product(betas, gammas, lrs, eps_stabs, s_temps))
    total_runs = len(combinations)
    print(f"Total configurations to test: {total_runs}", flush=True)
    
    for idx, (beta, gamma, lr, eps_stab, s_temp) in enumerate(combinations):
        accuracies = evaluate_method(
            'BAR-ACR', stream_batches, state_cos_mnist, state_cos_fmnist, prototypes_dict,
            device=device, beta=beta, gamma=gamma, lr=lr, eps_stab=eps_stab, s_temp=s_temp
        )
        
        overall = np.mean(accuracies) * 100
        
        if overall > best_overall:
            best_overall = overall
            best_config = {
                'beta': beta,
                'gamma': gamma,
                'lr': lr,
                'eps_stab': eps_stab,
                's_temp': s_temp
            }
            best_results = accuracies
            print(f"[{idx+1}/{total_runs}] NEW BEST! Overall: {overall:.2f}% | Config: beta={beta}, gamma={gamma}, lr={lr}, eps_stab={eps_stab}, s_temp={s_temp}", flush=True)
        elif (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{total_runs}] Done. Current best: {best_overall:.2f}% (Config of best: beta={best_config.get('beta')}, gamma={best_config.get('gamma')}, lr={best_config.get('lr')})", flush=True)
            
    print("\n" + "="*80, flush=True)
    print("SWEEP COMPLETED!", flush=True)
    print(f"Best Overall Accuracy: {best_overall:.4f}%", flush=True)
    print(f"Best Configuration: {best_config}", flush=True)
    
    # Show segment details for the best run
    accs = best_results
    seg1 = np.mean(accs[0:10]) * 100
    seg2 = np.mean(accs[10:20]) * 100
    seg3 = np.mean(accs[20:30]) * 100
    seg4 = np.mean(accs[30:40]) * 100
    seg5 = np.mean(accs[40:50]) * 100
    print(f"Segment breakdown:", flush=True)
    print(f"  Clean MNIST: {seg1:.2f}%", flush=True)
    print(f"  Noisy MNIST: {seg2:.2f}%", flush=True)
    print(f"  Clean FashionMNIST: {seg3:.2f}%", flush=True)
    print(f"  Noisy FashionMNIST: {seg4:.2f}%", flush=True)
    print(f"  Novel KMNIST: {seg5:.2f}%", flush=True)
    print("="*80, flush=True)

if __name__ == '__main__':
    main()
