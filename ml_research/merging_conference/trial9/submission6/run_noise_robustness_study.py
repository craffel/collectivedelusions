import os
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models import SimpleCNN
from run_experiments import ConfigurableHyperNet, manual_batch_norm2d, forward_merged, extract_batch_stats

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def add_noise(images, sigma):
    if sigma == 0.0:
        return images
    noise = torch.randn_like(images) * sigma
    return images + noise

def main():
    set_seed(42)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load expert models
    base_model = SimpleCNN().to(device)
    base_model.load_state_dict(torch.load("base_model.pth", map_location=device, weights_only=True))
    
    expert_mnist = SimpleCNN().to(device)
    expert_mnist.load_state_dict(torch.load("expert_mnist.pth", map_location=device, weights_only=True))
    
    expert_fashion = SimpleCNN().to(device)
    expert_fashion.load_state_dict(torch.load("expert_fashion.pth", map_location=device, weights_only=True))
    
    # Load hypernetwork
    hypernet = ConfigurableHyperNet(input_dim=278, hidden_layers=2, hidden_dim=128).to(device)
    hypernet.load_state_dict(torch.load("best_hypernet.pth", map_location=device, weights_only=True))
    
    # Freeze models
    for m in [base_model, expert_mnist, expert_fashion, hypernet]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading test dataset splits...")
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    sigmas = [0.0, 0.3, 0.6, 0.9, 1.2]
    methods = [
        "Expert MNIST Only",
        "Expert Fashion Only",
        "Uniform Merging (0.5/0.5)",
        "Gradient-based TTA (5 steps)",
        "Hyper-TTMM (Ours, Zero-Shot)"
    ]

    robustness_results = {sig: {m: [] for m in methods} for sig in sigmas}

    for sig in sigmas:
        print(f"\nEvaluating with Noise Level sigma = {sig}...")
        
        # Build test stream for this noise level
        mnist_eval_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
        fashion_eval_loader = DataLoader(fashion_test, batch_size=64, shuffle=False)
        
        eval_mnist_iter = iter(mnist_eval_loader)
        eval_fashion_iter = iter(fashion_eval_loader)
        
        test_stream = []
        for batch_idx in range(20):
            if batch_idx < 10:
                x, y = next(eval_mnist_iter)
            else:
                x, y = next(eval_fashion_iter)
            x = add_noise(x, sig)
            test_stream.append((x.to(device), y.to(device)))
            
        # Run methods
        for m_name in methods:
            accuracies = []
            for idx, (x, y) in enumerate(test_stream):
                if m_name == "Expert MNIST Only":
                    lambdas = torch.ones(6, device=device)
                    preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                elif m_name == "Expert Fashion Only":
                    lambdas = torch.zeros(6, device=device)
                    preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                elif m_name == "Uniform Merging (0.5/0.5)":
                    lambdas = torch.full((6,), 0.5, device=device)
                    preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                elif m_name == "Gradient-based TTA (5 steps)":
                    u = torch.zeros(6, requires_grad=True, device=device)
                    opt = optim.SGD([u], lr=0.05)
                    for step in range(5):
                        opt.zero_grad()
                        lmbds = torch.sigmoid(u)
                        outs = forward_merged(x, expert_mnist, expert_fashion, lmbds)
                        probs = F.softmax(outs, dim=1)
                        ent = - (probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                        ent.backward()
                        opt.step()
                    lambdas = torch.sigmoid(u).detach()
                    preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                elif m_name == "Hyper-TTMM (Ours, Zero-Shot)":
                    stats = extract_batch_stats(x, base_model, expert_mnist, expert_fashion)
                    with torch.no_grad():
                        lambdas = hypernet(stats.to(device))
                    preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
                _, pred_classes = torch.max(preds, 1)
                correct = (pred_classes == y).sum().item()
                acc = correct / len(y)
                accuracies.append(acc)
                
            mean_acc = np.mean(accuracies) * 100
            robustness_results[sig][m_name] = float(mean_acc)
            print(f"  {m_name:30s} : {mean_acc:.2f}%")

    with open("noise_robustness.json", "w", encoding="utf-8") as f:
        json.dump(robustness_results, f, indent=4)
    print("\nRobustness results saved to noise_robustness.json.")

if __name__ == "__main__":
    main()
