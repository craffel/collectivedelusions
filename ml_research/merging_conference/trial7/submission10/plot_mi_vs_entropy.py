import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Simple CNN structure matching run_experiment.py
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

def main():
    device = torch.device("cpu")
    print("Running plot_mi_vs_entropy.py on CPU...")
    
    # Load expert networks
    if not os.path.exists("checkpoints/mnist_expert.pt") or not os.path.exists("checkpoints/kmnist_expert.pt"):
        print("Error: Expert checkpoints not found in checkpoints/.")
        return
        
    mnist_expert = SimpleCNN()
    mnist_expert.load_state_dict(torch.load("checkpoints/mnist_expert.pt", map_location=device))
    mnist_expert.eval()
    
    kmnist_expert = SimpleCNN()
    kmnist_expert.load_state_dict(torch.load("checkpoints/kmnist_expert.pt", map_location=device))
    kmnist_expert.eval()
    
    experts = [mnist_expert, kmnist_expert]
    
    # Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="data", train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    
    batch_size = 64
    num_batches_per_domain = 10
    
    mnist_loader = DataLoader(Subset(mnist_test, list(range(num_batches_per_domain * batch_size))), batch_size=batch_size, shuffle=False)
    kmnist_loader = DataLoader(Subset(kmnist_test, list(range(num_batches_per_domain * batch_size))), batch_size=batch_size, shuffle=False)
    fmnist_loader = DataLoader(Subset(fmnist_test, list(range(num_batches_per_domain * batch_size))), batch_size=batch_size, shuffle=False)
    
    mnist_batches = list(mnist_loader)
    kmnist_batches = list(kmnist_loader)
    fmnist_batches = list(fmnist_loader)
    
    # Combine into a transition stream of 30 batches
    stream = []
    stream.extend(mnist_batches[:num_batches_per_domain])      # Batches 0-9: MNIST
    stream.extend(kmnist_batches[:num_batches_per_domain])     # Batches 10-19: KMNIST
    stream.extend(fmnist_batches[:num_batches_per_domain])     # Batches 20-29: FashionMNIST
    
    avg_entropies = []
    mi_uniforms = []
    mi_posteriors = []
    
    for idx, (x, _) in enumerate(stream):
        with torch.no_grad():
            p1 = F.softmax(mnist_expert(x), dim=-1)
            p2 = F.softmax(kmnist_expert(x), dim=-1)
            
            # Entropies per sample, then average over batch
            ent1 = -torch.sum(p1 * torch.log(p1 + 1e-10), dim=-1).mean().item()
            ent2 = -torch.sum(p2 * torch.log(p2 + 1e-10), dim=-1).mean().item()
            
            # 1. Average expert entropy
            avg_ent = (ent1 + ent2) / 2.0
            avg_entropies.append(avg_ent)
            
            # 2. MI with uniform weights
            p_uniform = 0.5 * p1 + 0.5 * p2
            ent_uniform = -torch.sum(p_uniform * torch.log(p_uniform + 1e-10), dim=-1).mean().item()
            mi_unif = ent_uniform - 0.5 * (ent1 + ent2)
            mi_uniforms.append(mi_unif)
            
            # 3. MI with soft posterior weights (gamma = 15)
            gamma = 15.0
            exp_neg_ent = np.exp(-gamma * np.array([ent1, ent2]))
            w = exp_neg_ent / np.sum(exp_neg_ent)
            p_soft = w[0] * p1 + w[1] * p2
            ent_soft = -torch.sum(p_soft * torch.log(p_soft + 1e-10), dim=-1).mean().item()
            mi_post = ent_soft - (w[0] * ent1 + w[1] * ent2)
            mi_posteriors.append(mi_post)
            
    # Create the plot
    plt.figure(figsize=(10, 5))
    
    # Draw vertical regions for domain boundaries
    plt.axvspan(0, 9.5, color='lightgreen', alpha=0.15, label='MNIST (Known)')
    plt.axvspan(9.5, 19.5, color='lightblue', alpha=0.15, label='KMNIST (Known)')
    plt.axvspan(19.5, 29, color='lightcoral', alpha=0.15, label='FashionMNIST (Novel)')
    
    # Plot curves
    plt.plot(avg_entropies, marker='o', linewidth=2.5, color='blue', label='Average Expert Entropy (Ours)')
    plt.plot(mi_uniforms, marker='s', linestyle='--', linewidth=2, color='red', label='Mutual Information (Uniform)')
    plt.plot(mi_posteriors, marker='^', linestyle=':', linewidth=2, color='green', label='Mutual Information (Posterior)')
    
    plt.xlabel('Test Stream Batch Index', fontsize=12)
    plt.ylabel('Uncertainty / Information Score', fontsize=12)
    plt.title('Novelty Detection Uncertainty Signals: Average Expert Entropy vs. Mutual Information', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(range(30))
    plt.xlim(-0.5, 29.5)
    plt.ylim(-0.1, 2.5)
    plt.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    # Annotate specific behaviors
    plt.text(4, 2.2, 'MNIST Expert active\nLow average entropy', color='darkgreen', ha='center', fontsize=9, bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))
    plt.text(14, 2.2, 'KMNIST Expert active\nLow average entropy', color='darkblue', ha='center', fontsize=9, bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))
    plt.text(24.5, 1.8, 'Novel domain:\nUniform consensus\nMI collapses,\nAverage Entropy peaks!', color='darkred', ha='center', fontsize=9, bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('mi_vs_entropy.png', dpi=300)
    print("Plot saved successfully as mi_vs_entropy.png.")
    
if __name__ == "__main__":
    main()
