import torch
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Import functions from original evaluate_ttmm
from evaluate_ttmm import SimpleCNN, evaluate_method, set_bn_mode, stable_softmax, get_distances, hoyer_sparsity, compute_entropy

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # Load expert models
    model_mnist = SimpleCNN().to(device)
    model_mnist.load_state_dict(torch.load("expert_mnist.pth", map_location=device))
    model_mnist.eval()
    
    model_fashion = SimpleCNN().to(device)
    model_fashion.load_state_dict(torch.load("expert_fashion.pth", map_location=device))
    model_fashion.eval()
    
    protos = torch.load("prototypes.pth", map_location=device)
    
    # Prepare datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    
    # Construct a transition stream of 11 batches of size 64
    # Batch 0: 100% MNIST (64 MNIST, 0 Fashion)
    # Batch 1: 90% MNIST (58 MNIST, 6 Fashion)
    # Batch 2: 80% MNIST (51 MNIST, 13 Fashion)
    # ...
    # Batch 10: 100% Fashion (0 MNIST, 64 Fashion)
    
    batches = []
    for i in range(11):
        ratio_mnist = 1.0 - (i * 0.1)
        num_mnist = int(round(64 * ratio_mnist))
        num_fashion = 64 - num_mnist
        
        mnist_indices = list(range(i * 64, i * 64 + num_mnist))
        fashion_indices = list(range(i * 64, i * 64 + num_fashion))
        
        x_mnist, y_mnist = [], []
        if num_mnist > 0:
            sub_mnist = Subset(mnist_test, mnist_indices)
            loader = DataLoader(sub_mnist, batch_size=num_mnist, shuffle=False)
            x_mnist, y_mnist = next(iter(loader))
            
        x_fashion, y_fashion = [], []
        if num_fashion > 0:
            sub_fashion = Subset(fmnist_test, fashion_indices)
            loader = DataLoader(sub_fashion, batch_size=num_fashion, shuffle=False)
            x_fashion, y_fashion = next(iter(loader))
            
        if num_mnist > 0 and num_fashion > 0:
            x_batch = torch.cat([x_mnist, x_fashion], dim=0)
            y_batch = torch.cat([y_mnist, y_fashion], dim=0)
        elif num_mnist > 0:
            x_batch, y_batch = x_mnist, y_mnist
        else:
            x_batch, y_batch = x_fashion, y_fashion
            
        # Optional: Add small amount of noise
        noise = torch.randn_like(x_batch) * 0.1
        x_batch_noisy = torch.clamp(x_batch + noise, -1.0, 1.0)
        
        batches.append((x_batch_noisy, y_batch, ratio_mnist))
        
    print(f"Generated {len(batches)} transition batches.")
    
    # Extract prototypes
    proto_mnist_norm = protos["mnist_norm"].to(device)
    proto_fashion_norm = protos["fashion_norm"].to(device)
    
    print("\nTransition Stream Routing Weights Analysis:")
    print("="*90)
    print(f"{'Batch':<5} | {'% MNIST':<8} | {'Hoyer Sparsity':<15} | {'BK-AHR w1':<12} | {'CSAIR w1':<12} | {'SCTS L2 w1_E':<12} | {'SCTS Ang w1_A':<12}")
    print("-"*90)
    
    for idx, (x_batch, y_batch, pct_mnist) in enumerate(batches):
        # 1. Hoyer sparsity
        x_pos = (x_batch + 1.0) / 2.0
        x_denoised = torch.where(x_pos > 0.35, x_pos, torch.zeros_like(x_pos))
        S_hoyer = hoyer_sparsity(x_denoised)
        
        # 2. Forward features
        set_bn_mode(model_mnist, train=True)
        set_bn_mode(model_fashion, train=True)
        with torch.no_grad():
            feats_mnist = model_mnist.forward_features(x_batch)
            feats_fashion = model_fashion.forward_features(x_batch)
            logits_mnist = model_mnist(x_batch)
            logits_fashion = model_fashion(x_batch)
            H_mnist = compute_entropy(logits_mnist).item()
            H_fashion = compute_entropy(logits_fashion).item()
            H_avg = 0.5 * (H_mnist + H_fashion)
            
        # BK-AHR with Normalized L2 routing
        # Euclidean
        feats_mnist_norm = feats_mnist / (feats_mnist.norm(p=2, dim=-1, keepdim=True) + 1e-9)
        feats_fashion_norm = feats_fashion / (feats_fashion.norm(p=2, dim=-1, keepdim=True) + 1e-9)
        d0_E = get_distances(feats_mnist_norm, proto_mnist_norm, "Euclidean")
        d1_E = get_distances(feats_fashion_norm, proto_fashion_norm, "Euclidean")
        D0_E, D1_E = d0_E.mean().item(), d1_E.mean().item()
        gap_E = abs(D0_E - D1_E)
        eps_stab_E = 0.08 / (1.0 + 5.0 * H_avg)
        tau_E = (gap_E / 3.0) + eps_stab_E
        w1_E, _ = stable_softmax(D0_E, D1_E, tau_E)
        
        # Angular
        d0_A = get_distances(feats_mnist, proto_mnist_norm, "Angular")
        d1_A = get_distances(feats_fashion, proto_fashion_norm, "Angular")
        D0_A, D1_A = d0_A.mean().item(), d1_A.mean().item()
        gap_A = abs(D0_A - D1_A)
        eps_stab_A = 0.04 / (1.0 + 5.0 * H_avg)
        tau_A = (gap_A / 3.0) + eps_stab_A
        w1_A, _ = stable_softmax(D0_A, D1_A, tau_A)
        
        # Method F (BK-AHR hard gate)
        if S_hoyer >= 0.50:
            w1_hard = w1_E
        else:
            w1_hard = w1_A
            
        # Method G (CSAIR soft sigmoid blend)
        lambda_blend = 1.0 / (1.0 + np.exp(-50.0 * (S_hoyer - 0.50)))
        w1_soft = lambda_blend * w1_E + (1.0 - lambda_blend) * w1_A
        
        print(f"{idx:<5} | {pct_mnist*100:>6.1f}% | {S_hoyer:<15.4f} | {w1_hard:<12.4f} | {w1_soft:<12.4f} | {w1_E:<12.4f} | {w1_A:<12.4f}")

if __name__ == "__main__":
    main()
