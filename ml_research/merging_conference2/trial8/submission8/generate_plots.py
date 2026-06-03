import os
import copy
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define noise classes
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Loader constructor
def get_subset_test_loader(task_name, std=0.0, limit=2000):
    t_list = []
    if task_name in ["mnist", "fmnist"]:
        t_list.append(transforms.Resize((32, 32)))
        t_list.append(transforms.ToTensor())
        t_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    else:
        t_list.append(transforms.ToTensor())
        
    t_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    if std > 0.0:
        t_list.append(AddGaussianNoise(0.0, std))
        
    transform = transforms.Compose(t_list)
    
    if task_name == "mnist":
        ds = MNIST(root="./data", train=False, download=True, transform=transform)
    elif task_name == "fmnist":
        ds = FashionMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        ds = CIFAR10(root="./data", train=False, download=True, transform=transform)
        
    # Take a deterministic subset for speed
    indices = list(range(min(limit, len(ds))))
    subset_ds = Subset(ds, indices)
    return DataLoader(subset_ds, batch_size=256, shuffle=False, num_workers=2)

def load_model_from_checkpoint(path):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

# Reuse weight calculation functions
from merge_and_evaluate import compute_uipr_weights, compute_ucpc_weights, compute_rcpc_weights, merge_batch_norms

def main():
    # 1. Load checkpoints
    print("\n--- LOADING MODELS ---")
    progenitor = load_model_from_checkpoint("checkpoints/progenitor.pth")
    expert_mnist = load_model_from_checkpoint("checkpoints/expert_mnist.pth")
    expert_fmnist = load_model_from_checkpoint("checkpoints/expert_fmnist.pth")
    expert_cifar10 = load_model_from_checkpoint("checkpoints/expert_cifar10.pth")

    experts = [expert_mnist, expert_fmnist, expert_cifar10]
    expert_names = ["mnist", "fmnist", "cifar10"]
    K = len(experts)

    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]

    # Pre-merge backbone weights
    print("Preparing standard merged (WA) state...")
    wa_state = copy.deepcopy(progenitor_state)
    for key in progenitor_state.keys():
        if not key.startswith("fc."):
            tensors = [expert_states[k][key] for k in range(K)]
            wa_state[key] = sum(tensors) / K

    # Compute calibration states (using average BNs for unified deployment)
    print("Computing calibration weight states...")
    uipr_state = compute_uipr_weights(progenitor_state, expert_states, wa_state)
    ucpc_state = compute_ucpc_weights(progenitor_state, expert_states, wa_state, version="v1")
    merged_bn = merge_batch_norms(expert_states)

    # ==========================================
    # SWEEP 1: STOCHASTIC RESONANCE ON MNIST
    # ==========================================
    print("\n--- SWEEP 1: STOCHASTIC RESONANCE ---")
    noise_stds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    wa_mnist_accs = []
    uipr_mnist_accs = []
    ucpc_mnist_accs = []
    
    # We will evaluate on a subset of 2000 images for speed on CPU
    for std in noise_stds:
        print(f"Evaluating MNIST noise std = {std:.2f}...")
        loader = get_subset_test_loader("mnist", std=std, limit=2000)
        
        # 1. Weight Averaging
        state = copy.deepcopy(wa_state)
        state.update(merged_bn)
        state.update({k: v for k, v in expert_states[0].items() if k.startswith("fc.")})
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(state)
        model = model.to(device)
        wa_mnist_accs.append(evaluate_model(model, loader))
        
        # 2. U-IPR
        state = copy.deepcopy(uipr_state)
        state.update(merged_bn)
        state.update({k: v for k, v in expert_states[0].items() if k.startswith("fc.")})
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(state)
        model = model.to(device)
        uipr_mnist_accs.append(evaluate_model(model, loader))
        
        # 3. UCPC
        state = copy.deepcopy(ucpc_state)
        state.update(merged_bn)
        state.update({k: v for k, v in expert_states[0].items() if k.startswith("fc.")})
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(state)
        model = model.to(device)
        ucpc_mnist_accs.append(evaluate_model(model, loader))

    # Plot Stochastic Resonance
    plt.figure(figsize=(7, 5))
    plt.plot(noise_stds, wa_mnist_accs, 'o-', label="Weight Averaging", color="#d62728", linewidth=2, markersize=6)
    plt.plot(noise_stds, uipr_mnist_accs, 's-', label="U-IPR (Layer-wise)", color="#1f77b4", linewidth=2, markersize=6)
    plt.plot(noise_stds, ucpc_mnist_accs, '^-', label="UCPC (Ours - Channel-wise)", color="#2ca02c", linewidth=2, markersize=6)
    
    plt.title("Stochastic Resonance under Scale Collapse (MNIST)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Input Gaussian Noise Std Dev ($\\sigma$)", fontsize=11, labelpad=8)
    plt.ylabel("Classification Accuracy (%)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig("stochastic_resonance.pdf", dpi=300)
    plt.savefig("stochastic_resonance.png", dpi=300)
    plt.close()
    print("Saved stochastic_resonance.pdf and .png")

    # ==========================================
    # SWEEP 2: RCPC PARETO FRONTIER
    # ==========================================
    print("\n--- SWEEP 2: RCPC PARETO FRONTIER ---")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Pre-build loaders for clean data (limit=1000 for speed)
    clean_loaders = {name: get_subset_test_loader(name, std=0.0, limit=1000) for name in expert_names}
    
    rcpc_avg_accs = []
    
    for alpha in alphas:
        print(f"Evaluating RCPC with alpha = {alpha:.1f}...")
        rcpc_state = compute_rcpc_weights(progenitor_state, expert_states, wa_state, alpha=alpha, version="v2")
        
        accs = []
        for idx, task_name in enumerate(expert_names):
            state = copy.deepcopy(rcpc_state)
            state.update(merged_bn)
            state.update({k: v for k, v in expert_states[idx].items() if k.startswith("fc.")})
            
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(512, 10)
            model.load_state_dict(state)
            model = model.to(device)
            accs.append(evaluate_model(model, clean_loaders[task_name]))
            
        rcpc_avg_accs.append(np.mean(accs))

    # Plot RCPC Pareto Frontier
    plt.figure(figsize=(7, 5))
    plt.plot(alphas, rcpc_avg_accs, 'D-', color="#8c564b", linewidth=2.5, markersize=7, label="RCPC")
    
    # Annotate specific endpoints
    plt.annotate("U-IPR (Layer-wise)\n$\\alpha=0.0$", xy=(0.0, rcpc_avg_accs[0]), xytext=(0.05, rcpc_avg_accs[0] - 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, headlength=6),
                 fontsize=9, fontweight='bold')
    plt.annotate("UCPC (Ours)\n$\\alpha=1.0$", xy=(1.0, rcpc_avg_accs[-1]), xytext=(0.75, rcpc_avg_accs[-1] - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, headlength=6),
                 fontsize=9, fontweight='bold', color="#2ca02c")

    plt.title("RCPC Pareto Frontier (Granularity Blending)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Blending Coefficient $\\alpha$ (0.0=Layer, 1.0=Channel)", fontsize=11, labelpad=8)
    plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(min(rcpc_avg_accs) - 0.5, max(rcpc_avg_accs) + 0.5)
    plt.tight_layout()
    plt.savefig("rcpc_pareto.pdf", dpi=300)
    plt.savefig("rcpc_pareto.png", dpi=300)
    plt.close()
    print("Saved rcpc_pareto.pdf and .png")

if __name__ == "__main__":
    main()
