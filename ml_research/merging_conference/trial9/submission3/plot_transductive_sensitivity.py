import torch
import numpy as np
import matplotlib.pyplot as plt
from eval_differentiable import evaluate_differentiable
import data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running transductive sensitivity analysis on device: {device}")
    
    # Load experts and data
    state_cos_mnist = torch.load('checkpoints/cos_mnist.pt', map_location=device)
    state_cos_fmnist = torch.load('checkpoints/cos_fmnist.pt', map_location=device)
    prototypes_dict = torch.load('checkpoints/prototypes.pt', map_location=device)
    
    mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test = data.get_datasets()
    stream_batches = data.create_non_stationary_stream(mnist_test, fmnist_test, kmnist_test)
    
    # 1. Sweep beta (Prior KL Penalty)
    # Keeping defaults: gamma=0.0001, lr=0.005, train_mode=True
    betas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]
    beta_accuracies = []
    
    print("\nSweeping beta (transductive)...")
    for beta in betas:
        accuracies = evaluate_differentiable(
            'BAR-ACR', stream_batches, state_cos_mnist, state_cos_fmnist, prototypes_dict,
            device=device, beta=beta, gamma=0.0001, lr=0.005, eps_stab=0.001, s_temp=3.5,
            use_soft_bn=True, use_precond=True, num_steps=5, train_mode=True
        )
        overall = np.mean(accuracies) * 100
        beta_accuracies.append(overall)
        print(f"  beta={beta:.2f} -> Overall Accuracy: {overall:.2f}%")
        
    # 2. Sweep gamma (Coherence Penalty)
    # Keeping defaults: beta=0.2, lr=0.005, train_mode=True
    gammas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    gamma_accuracies = []
    
    print("\nSweeping gamma (transductive)...")
    for gamma in gammas:
        accuracies = evaluate_differentiable(
            'BAR-ACR', stream_batches, state_cos_mnist, state_cos_fmnist, prototypes_dict,
            device=device, beta=0.2, gamma=gamma, lr=0.005, eps_stab=0.001, s_temp=3.5,
            use_soft_bn=True, use_precond=True, num_steps=5, train_mode=True
        )
        overall = np.mean(accuracies) * 100
        gamma_accuracies.append(overall)
        print(f"  gamma={gamma} -> Overall Accuracy: {overall:.2f}%")
        
    # 3. Sweep learning rate (lr)
    # Keeping defaults: beta=0.2, gamma=0.0001, train_mode=True
    lrs = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    lr_accuracies = []
    
    print("\nSweeping lr (transductive)...")
    for lr in lrs:
        accuracies = evaluate_differentiable(
            'BAR-ACR', stream_batches, state_cos_mnist, state_cos_fmnist, prototypes_dict,
            device=device, beta=0.2, gamma=0.0001, lr=lr, eps_stab=0.001, s_temp=3.5,
            use_soft_bn=True, use_precond=True, num_steps=5, train_mode=True
        )
        overall = np.mean(accuracies) * 100
        lr_accuracies.append(overall)
        print(f"  lr={lr} -> Overall Accuracy: {overall:.2f}%")
        
    # Set up matplotlib style for academic quality
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Beta Sensitivity
    ax1.plot(betas, beta_accuracies, marker='o', color='#1f77b4', linewidth=2, markersize=6, label='BAR-ACR')
    ax1.set_xlabel(r'Prior KL Penalty ($\beta$)')
    ax1.set_ylabel('Overall Accuracy (%)')
    ax1.set_title(r'Sensitivity to Prior KL Penalty $\beta$')
    ax1.grid(True)
    ax1.axvline(x=0.2, color='r', linestyle='--', label='Optimal Default (0.2)')
    ax1.legend()
    
    # Plot 2: Gamma Sensitivity
    gamma_labels = [str(g) if g > 0 else '0' for g in gammas]
    ax2.plot(range(len(gammas)), gamma_accuracies, marker='s', color='#2ca02c', linewidth=2, markersize=6, label='BAR-ACR')
    ax2.set_xticks(range(len(gammas)))
    ax2.set_xticklabels(gamma_labels)
    ax2.set_xlabel(r'Coherence Penalty ($\gamma$)')
    ax2.set_ylabel('Overall Accuracy (%)')
    ax2.set_title(r'Sensitivity to Coherence Penalty $\gamma$')
    ax2.grid(True)
    ax2.axvline(x=3, color='r', linestyle='--', label='Optimal Default (1e-4)') # index 3 corresponds to 1e-4
    ax2.legend()
    
    # Plot 3: LR Sensitivity
    ax3.plot(lrs, lr_accuracies, marker='^', color='#ff7f0e', linewidth=2, markersize=6, label='BAR-ACR')
    ax3.set_xlabel(r'Learning Rate ($\eta$)')
    ax3.set_ylabel('Overall Accuracy (%)')
    ax3.set_title(r'Sensitivity to Learning Rate $\eta$')
    ax3.grid(True)
    ax3.axvline(x=0.005, color='r', linestyle='--', label='Optimal Default (0.005)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('transductive_sensitivity.png', dpi=300)
    print("\nTransductive sensitivity plot saved as transductive_sensitivity.png")

if __name__ == '__main__':
    main()
