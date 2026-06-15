import os
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

device = torch.device("cpu") # Keep on CPU for simplicity and speed

# 1. Dataset Loading (Native 28x28, fast on CPU)
def get_loaders(num_train=1000, num_test=200, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # MNIST
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # FashionMNIST
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Subsets
    mnist_train_sub = Subset(mnist_train, list(range(num_train)))
    mnist_test_sub = Subset(mnist_test, list(range(num_test)))
    fmnist_train_sub = Subset(fmnist_train, list(range(num_train)))
    fmnist_test_sub = Subset(fmnist_test, list(range(num_test)))
    
    loaders = {
        'MNIST': {
            'train': DataLoader(mnist_train_sub, batch_size=batch_size, shuffle=True),
            'test': DataLoader(mnist_test_sub, batch_size=batch_size, shuffle=False)
        },
        'FashionMNIST': {
            'train': DataLoader(fmnist_train_sub, batch_size=batch_size, shuffle=True),
            'test': DataLoader(fmnist_test_sub, batch_size=batch_size, shuffle=False)
        }
    }
    return loaders

# 2. Define simple 3-layer MLP without normalization
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x, record_activations=False):
        x = x.view(-1, 28*28)
        
        act1 = self.fc1(x)
        x1 = self.relu(act1)
        
        act2 = self.fc2(x1)
        x2 = self.relu(act2)
        
        out = self.fc3(x2)
        
        if record_activations:
            return out, {
                'act1_norm': torch.linalg.norm(act1, ord='fro').item() / x.size(0),
                'act2_norm': torch.linalg.norm(act2, ord='fro').item() / x.size(0)
            }
        return out

# 3. Training Function
def train_model(model, loader, epochs=3, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 4. Evaluation Function
def evaluate_model(model, loader, record_activations=False):
    model.eval()
    correct = 0
    total = 0
    act1_norms = []
    act2_norms = []
    
    with torch.no_grad():
        for images, labels in loader:
            if record_activations:
                outputs, acts = model(images, record_activations=True)
                act1_norms.append(acts['act1_norm'])
                act2_norms.append(acts['act2_norm'])
            else:
                outputs = model(images)
                
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    if record_activations:
        return acc, np.mean(act1_norms), np.mean(act2_norms)
    return acc

# 5. Merging Operators
def apply_bwn_to_param(merged_p, expert_ps, lambdas):
    if merged_p.dim() >= 2:
        expert_norms = [torch.linalg.norm(ep, ord='fro') for ep in expert_ps]
        merged_norm = torch.linalg.norm(merged_p, ord='fro')
    else:
        expert_norms = [torch.linalg.norm(ep) for ep in expert_ps]
        merged_norm = torch.linalg.norm(merged_p)
        
    total_lambda = sum(lambdas)
    if total_lambda == 0:
        weights = [1.0 / len(lambdas)] * len(lambdas)
    else:
        weights = [l / total_lambda for l in lambdas]
        
    target_norm = sum(w * norm for w, norm in zip(weights, expert_norms))
    alpha = target_norm / torch.clamp(merged_norm, min=1e-8)
    return alpha * merged_p

def merge_models(base_model, experts, lambdas, rank=None, apply_bwn=False):
    merged_model = copy.deepcopy(base_model)
    
    # List of layer names to merge
    layer_names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
    
    base_state = base_model.state_dict()
    expert_states = [exp.state_dict() for exp in experts]
    merged_state = merged_model.state_dict()
    
    for name in layer_names:
        W0 = base_state[name]
        W_experts = [state[name] for state in expert_states]
        
        # Calculate task vectors
        T_experts = [W_exp - W0 for W_exp in W_experts]
        
        # If rank is specified and it's a 2D weight matrix, apply SVS
        if rank is not None and 'weight' in name:
            T_processed = []
            for T in T_experts:
                U, S, Vh = torch.linalg.svd(T, full_matrices=False)
                # Slice to rank k
                k = min(rank, len(S))
                U_k = U[:, :k]
                S_k = torch.diag(S[:k])
                Vh_k = Vh[:k, :]
                T_sliced = U_k @ S_k @ Vh_k
                T_processed.append(T_sliced)
        else:
            T_processed = T_experts
            
        # Linear combination
        W_merged_delta = sum(l * T for l, T in zip(lambdas, T_processed))
        W_merged = W0 + W_merged_delta
        
        # Apply BWN if enabled
        if apply_bwn:
            W_merged = apply_bwn_to_param(W_merged, W_experts, lambdas)
            
        merged_state[name].copy_(W_merged)
        
    merged_model.load_state_dict(merged_state)
    return merged_model

def main():
    print("Running MLP + BWN Ablation Experiments in Non-Normalized Environment...")
    loaders = get_loaders()
    
    # Initialize shared base model
    base_model = SimpleMLP()
    
    # Train Expert A on MNIST
    expert_a = copy.deepcopy(base_model)
    print("Training Expert A on MNIST...")
    train_model(expert_a, loaders['MNIST']['train'], epochs=3)
    acc_a = evaluate_model(expert_a, loaders['MNIST']['test'])
    print(f"Expert A MNIST Test Acc: {acc_a:.2f}%")
    
    # Train Expert B on FashionMNIST
    expert_b = copy.deepcopy(base_model)
    print("Training Expert B on FashionMNIST...")
    train_model(expert_b, loaders['FashionMNIST']['train'], epochs=3)
    acc_b = evaluate_model(expert_b, loaders['FashionMNIST']['test'])
    print(f"Expert B FashionMNIST Test Acc: {acc_b:.2f}%")
    
    # Baseline Expert accuracies
    print("\nEvaluating individual experts on other tasks (interference check):")
    print(f"Expert A on FashionMNIST: {evaluate_model(expert_a, loaders['FashionMNIST']['test'])}%")
    print(f"Expert B on MNIST: {evaluate_model(expert_b, loaders['MNIST']['test'])}%")
    
    experts = [expert_a, expert_b]
    
    # Let's sweep lambda for Task Arithmetic (with and without BWN)
    lambda_sweep = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    ta_nobwn_accs = []
    ta_bwn_accs = []
    
    ta_nobwn_act1 = []
    ta_bwn_act1 = []
    
    for l in lambda_sweep:
        lambdas = [l, l]
        
        # TA without BWN
        model_ta_nobwn = merge_models(base_model, experts, lambdas, apply_bwn=False)
        acc_mnist = evaluate_model(model_ta_nobwn, loaders['MNIST']['test'])
        acc_fmnist, act1, act2 = evaluate_model(model_ta_nobwn, loaders['FashionMNIST']['test'], record_activations=True)
        avg_acc = (acc_mnist + acc_fmnist) / 2.0
        ta_nobwn_accs.append(avg_acc)
        ta_nobwn_act1.append(act1)
        
        # TA with BWN
        model_ta_bwn = merge_models(base_model, experts, lambdas, apply_bwn=True)
        acc_mnist_bwn = evaluate_model(model_ta_bwn, loaders['MNIST']['test'])
        acc_fmnist_bwn, act1_bwn, act2_bwn = evaluate_model(model_ta_bwn, loaders['FashionMNIST']['test'], record_activations=True)
        avg_acc_bwn = (acc_mnist_bwn + acc_fmnist_bwn) / 2.0
        ta_bwn_accs.append(avg_acc_bwn)
        ta_bwn_act1.append(act1_bwn)
        
        print(f"Lambda={l:.1f} | TA No BWN Acc: {avg_acc:.2f}% (Act1 Norm: {act1:.4f}) | TA + BWN Acc: {avg_acc_bwn:.2f}% (Act1 Norm: {act1_bwn:.4f})")
        
    # Let's sweep SVS ranks at a fixed lambda (say l=0.5)
    ranks = [4, 8, 16, 32, 64]
    svs_nobwn_accs = []
    svs_bwn_accs = []
    
    l_fixed = 0.5
    lambdas_fixed = [l_fixed, l_fixed]
    print(f"\nEvaluating SVS with and without BWN across ranks $k$ at lambda={l_fixed}:")
    for r in ranks:
        # SVS without BWN
        model_svs_nobwn = merge_models(base_model, experts, lambdas_fixed, rank=r, apply_bwn=False)
        acc_mnist = evaluate_model(model_svs_nobwn, loaders['MNIST']['test'])
        acc_fmnist, act1, act2 = evaluate_model(model_svs_nobwn, loaders['FashionMNIST']['test'], record_activations=True)
        avg_acc = (acc_mnist + acc_fmnist) / 2.0
        svs_nobwn_accs.append(avg_acc)
        
        # SVS with BWN
        model_svs_bwn = merge_models(base_model, experts, lambdas_fixed, rank=r, apply_bwn=True)
        acc_mnist_bwn = evaluate_model(model_svs_bwn, loaders['MNIST']['test'])
        acc_fmnist_bwn, act1_bwn, act2_bwn = evaluate_model(model_svs_bwn, loaders['FashionMNIST']['test'], record_activations=True)
        avg_acc_bwn = (acc_mnist_bwn + acc_fmnist_bwn) / 2.0
        svs_bwn_accs.append(avg_acc_bwn)
        
        print(f"Rank={r:2d} | SVS No BWN Acc: {avg_acc:.2f}% (Act1 Norm: {act1:.4f}) | SVS + BWN Acc: {avg_acc_bwn:.2f}% (Act1 Norm: {act1_bwn:.4f})")
        
    # Plotting Figure 4: MLP + BWN Ablation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Accuracy vs Lambda
    ax1.plot(lambda_sweep, ta_nobwn_accs, label="Task Arithmetic (TA) No BWN", color='red', marker='o', linestyle='--')
    ax1.plot(lambda_sweep, ta_bwn_accs, label="Task Arithmetic (TA) + BWN", color='blue', marker='s')
    ax1.set_title("Average Multi-Task Accuracy vs. Scaling $\lambda$", fontsize=12)
    ax1.set_xlabel("Scaling Coefficient $\lambda$", fontsize=10)
    ax1.set_ylabel("Average Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9)
    
    # 2. Activation Norms vs Lambda
    ax2.plot(lambda_sweep, ta_nobwn_act1, label="TA No BWN (Layer 1 Activation Scale)", color='red', marker='o', linestyle='--')
    ax2.plot(lambda_sweep, ta_bwn_act1, label="TA + BWN (Layer 1 Activation Scale)", color='blue', marker='s')
    ax2.set_title("Activation Scale vs. Scaling $\lambda$", fontsize=12)
    ax2.set_xlabel("Scaling Coefficient $\lambda$", fontsize=10)
    ax2.set_ylabel("Frobenius Norm of Layer 1 Activations", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9)
    
    plt.suptitle("BWN Validation in Non-Normalized MLP Architecture (MNIST/FashionMNIST)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig("results/fig4_mlp_bwn_ablation.png", dpi=150)
    plt.close()
    
    # Save MLP results to JSON
    mlp_results = {
        'lambda_sweep': lambda_sweep,
        'ta_nobwn_accs': ta_nobwn_accs,
        'ta_bwn_accs': ta_bwn_accs,
        'ta_nobwn_act1': ta_nobwn_act1,
        'ta_bwn_act1': ta_bwn_act1,
        'ranks': ranks,
        'svs_nobwn_accs': svs_nobwn_accs,
        'svs_bwn_accs': svs_bwn_accs
    }
    with open("results/mlp_metrics_summary.json", "w") as f:
        json.dump(mlp_results, f, indent=2)
        
    print("\nMLP BWN experiment complete. Saved figures to results/fig4_mlp_bwn_ablation.png and metrics to results/mlp_metrics_summary.json")

if __name__ == "__main__":
    main()
