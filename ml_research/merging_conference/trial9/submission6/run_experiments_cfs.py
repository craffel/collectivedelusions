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
from torch.utils.data import DataLoader
from models import SimpleCNN
from run_experiments import ConfigurableHyperNet, manual_batch_norm2d, forward_merged, extract_batch_stats

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def add_noise(images, sigma):
    if sigma == 0.0:
        return images
    noise = torch.randn_like(images) * sigma
    return images + noise

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, transform=transform, download=True)

    mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True, drop_last=True)
    fashion_test_loader = DataLoader(fashion_test, batch_size=64, shuffle=True, drop_last=True)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True, drop_last=True)

    # Infinite iterators helper
    def get_infinite_iter(loader):
        while True:
            for x, y in loader:
                yield x, y

    mnist_iter = get_infinite_iter(mnist_test_loader)
    fashion_iter = get_infinite_iter(fashion_test_loader)
    kmnist_iter = get_infinite_iter(kmnist_test_loader)

    # 1. Build Continuously Fluctuating Stream (CFS) of 50 batches
    print("\n=== Constructing Continuously Fluctuating Stream (CFS) ===")
    test_stream = []
    stream_metadata = [] # stores (p, sigma, domain_label)
    
    for t in range(50):
        if t < 25:
            # Gradual shift: starts mostly MNIST, oscillates
            p = 0.5 + 0.4 * np.cos(np.pi * t / 12.0)
            sigma = 0.3 * (1.0 - np.cos(np.pi * t / 12.0))
            domain = "gradual_mnist"
            
            # Mix batch
            xm, ym = next(mnist_iter)
            xf, yf = next(fashion_iter)
            num_m = int(round(64 * p))
            num_f = 64 - num_m
            x = torch.cat([xm[:num_m], xf[:num_f]], dim=0)
            y = torch.cat([ym[:num_m], yf[:num_f]], dim=0)
            x = add_noise(x, sigma)
            
        elif t < 40:
            # Sudden domain shift: at t=25, jump to Fashion-heavy, then gradual shift
            p = 0.5 - 0.4 * np.cos(np.pi * (t - 25) / 7.5)
            sigma = 0.3 * (1.0 + np.cos(np.pi * (t - 25) / 7.5))
            domain = "sudden_fashion"
            
            xm, ym = next(mnist_iter)
            xf, yf = next(fashion_iter)
            num_m = int(round(64 * p))
            num_f = 64 - num_m
            x = torch.cat([xm[:num_m], xf[:num_f]], dim=0)
            y = torch.cat([ym[:num_m], yf[:num_f]], dim=0)
            x = add_noise(x, sigma)
            
        else:
            # Sudden shift to 100% OOD KMNIST
            p = 0.0
            sigma = 0.0
            domain = "ood_kmnist"
            
            x, y = next(kmnist_iter)
            
        test_stream.append((x.to(device), y.to(device)))
        stream_metadata.append({"p": float(p), "sigma": float(sigma), "domain": domain})
        
    print(f"CFS built successfully with 50 batches.")

    methods = [
        "Expert MNIST Only",
        "Expert Fashion Only",
        "Uniform Merging (0.5/0.5)",
        "Oracle Merging (Ceiling)",
        "Gradient-based TTA (5 steps)",
        "Hyper-TTMM (Ours, Zero-Shot)",
        "Hyper-TTMM + Naive EMA (alpha=0.9)",
        "Hyper-TTMM + Adaptive EMA (alpha=0.5)"
    ]

    results = {m: {"accuracies": [], "latencies": [], "phase_accs": {"gradual_mnist": [], "sudden_fashion": [], "ood_kmnist": []}} for m in methods}

    print("\n=== Evaluating on Continuously Fluctuating Stream ===")
    for m_name in methods:
        print(f"Evaluating: {m_name}...")
        
        # State variables for EMA
        smoothed_stats = None
        prev_raw_stats = None
        tau = 0.3 # Threshold for Adaptive EMA
        
        for idx, (x, y) in enumerate(test_stream):
            meta = stream_metadata[idx]
            domain = meta["domain"]
            
            start_time = time.perf_counter()
            
            if m_name == "Expert MNIST Only":
                lambdas = torch.ones(6, device=device)
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
            elif m_name == "Expert Fashion Only":
                lambdas = torch.zeros(6, device=device)
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
            elif m_name == "Uniform Merging (0.5/0.5)":
                lambdas = torch.full((6,), 0.5, device=device)
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
            elif m_name == "Oracle Merging (Ceiling)":
                u = torch.zeros(6, requires_grad=True, device=device)
                opt = optim.Adam([u], lr=0.1)
                criterion = nn.CrossEntropyLoss()
                for step in range(30):
                    opt.zero_grad()
                    lmbds = torch.sigmoid(u)
                    outs = forward_merged(x, expert_mnist, expert_fashion, lmbds)
                    loss = criterion(outs, y)
                    loss.backward()
                    opt.step()
                lambdas = torch.sigmoid(u).detach()
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
                
            elif m_name == "Hyper-TTMM + Naive EMA (alpha=0.9)":
                raw_stats = extract_batch_stats(x, base_model, expert_mnist, expert_fashion)
                if smoothed_stats is None:
                    smoothed_stats = raw_stats.clone()
                else:
                    smoothed_stats = 0.9 * smoothed_stats + 0.1 * raw_stats
                with torch.no_grad():
                    lambdas = hypernet(smoothed_stats.to(device))
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
            elif m_name == "Hyper-TTMM + Adaptive EMA (alpha=0.5)":
                raw_stats = extract_batch_stats(x, base_model, expert_mnist, expert_fashion)
                
                # Check for reset trigger using consecutive raw batch descriptors distance
                if prev_raw_stats is not None:
                    dist = torch.norm(raw_stats - prev_raw_stats, p=2).item()
                    if dist > tau:
                        # Reset smoothed stats to current raw stats instantly!
                        smoothed_stats = raw_stats.clone()
                    else:
                        smoothed_stats = 0.5 * smoothed_stats + 0.5 * raw_stats
                else:
                    smoothed_stats = raw_stats.clone()
                
                prev_raw_stats = raw_stats.clone()
                
                with torch.no_grad():
                    lambdas = hypernet(smoothed_stats.to(device))
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)

            latency = time.perf_counter() - start_time
            _, pred_classes = torch.max(preds, 1)
            correct = (pred_classes == y).sum().item()
            acc = correct / len(y)
            
            results[m_name]["accuracies"].append(acc)
            results[m_name]["latencies"].append(latency)
            results[m_name]["phase_accs"][domain].append(acc)

        all_accs = results[m_name]["accuracies"]
        avg_acc = np.mean(all_accs) * 100
        avg_lat = np.mean(results[m_name]["latencies"]) * 1000
        print(f"-> Overall: {avg_acc:.2f}% | Latency: {avg_lat:.2f}ms")
        for ph_name, ph_list in results[m_name]["phase_accs"].items():
            ph_acc = np.mean(ph_list) * 100
            print(f"   Phase {ph_name:15s} Acc: {ph_acc:.2f}%")

    # Save results to json
    cfs_results = {}
    for m in methods:
        cfs_results[m] = {
            "overall_accuracy": float(np.mean(results[m]["accuracies"]) * 100),
            "average_latency_ms": float(np.mean(results[m]["latencies"]) * 1000),
            "phase_accuracies": {
                k: float(np.mean(v) * 100) for k, v in results[m]["phase_accs"].items()
            }
        }

    with open("results_cfs.json", "w", encoding="utf-8") as f:
        json.dump(cfs_results, f, indent=4)
    print("\nCFS results saved to results_cfs.json.")

if __name__ == "__main__":
    main()
