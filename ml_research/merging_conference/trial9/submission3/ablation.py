import torch
import numpy as np
import os
from evaluate import evaluate_method
import data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Ablation Study on device: {device}")
    
    # Load experts and prototypes
    state_cos_mnist = torch.load('checkpoints/cos_mnist.pt', map_location=device)
    state_cos_fmnist = torch.load('checkpoints/cos_fmnist.pt', map_location=device)
    prototypes_dict = torch.load('checkpoints/prototypes.pt', map_location=device)
    
    # Load stream
    mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test = data.get_datasets()
    stream_batches = data.create_non_stationary_stream(mnist_test, fmnist_test, kmnist_test)
    
    # Baseline hyperparameters
    hparams = {
        'beta': 0.2,
        'gamma': 0.0001,
        'lr': 0.005,
        'eps_stab': 0.001,
        's_temp': 3.5
    }
    
    # Define ablation variants
    variants = {
        'Full BAR-ACR (Ours)': {
            'use_soft_bn': True,
            'use_precond': True,
            **hparams
        },
        'w/o Soft BN Buffer Fusion': {
            'use_soft_bn': False,
            'use_precond': True,
            **hparams
        },
        'w/o Coherence Regularization (gamma=0)': {
            'use_soft_bn': True,
            'use_precond': True,
            **hparams,
            'gamma': 0.0
        },
        'w/o Prior KL Regularization (beta=0)': {
            'use_soft_bn': True,
            'use_precond': True,
            **hparams,
            'beta': 0.0
        },
        'w/o Gradient Preconditioning': {
            'use_soft_bn': True,
            'use_precond': False,
            **hparams
        },
        'w/o Test-Time Parameter Update (lr=0)': {
            'use_soft_bn': True,
            'use_precond': True,
            **hparams,
            'lr': 0.0
        }
    }
    
    results = {}
    
    for name, config in variants.items():
        print(f"\nRunning variant: {name}")
        accuracies = evaluate_method(
            'BAR-ACR',
            stream_batches,
            state_cos_mnist,
            state_cos_fmnist,
            prototypes_dict,
            device=device,
            beta=config['beta'],
            gamma=config['gamma'],
            lr=config['lr'],
            eps_stab=config['eps_stab'],
            s_temp=config['s_temp'],
            use_soft_bn=config['use_soft_bn'],
            use_precond=config['use_precond']
        )
        
        results[name] = accuracies
        
        # Breakdown
        seg1 = np.mean(accuracies[0:10]) * 100
        seg2 = np.mean(accuracies[10:20]) * 100
        seg3 = np.mean(accuracies[20:30]) * 100
        seg4 = np.mean(accuracies[30:40]) * 100
        seg5 = np.mean(accuracies[40:50]) * 100
        overall = np.mean(accuracies) * 100
        
        print(f"  Clean MNIST: {seg1:.2f}% | Noisy MNIST: {seg2:.2f}% | Clean Fashion: {seg3:.2f}% | Noisy Fashion: {seg4:.2f}% | Novel KMNIST: {seg5:.2f}% | Overall: {overall:.2f}%")
        
    print("\n" + "="*90)
    print(f"{'Variant':<40} | {'C-MN':<7} | {'N-MN':<7} | {'C-FN':<7} | {'N-FN':<7} | {'Nov-K':<7} | {'Overall':<7}")
    print("-"*90)
    for name in variants:
        accs = results[name]
        seg1 = np.mean(accs[0:10]) * 100
        seg2 = np.mean(accs[10:20]) * 100
        seg3 = np.mean(accs[20:30]) * 100
        seg4 = np.mean(accs[30:40]) * 100
        seg5 = np.mean(accs[40:50]) * 100
        overall = np.mean(accs) * 100
        print(f"{name:<40} | {seg1:5.2f}% | {seg2:5.2f}% | {seg3:5.2f}% | {seg4:5.2f}% | {seg5:5.2f}% | {overall:5.2f}%")
    print("="*90)

if __name__ == '__main__':
    main()
