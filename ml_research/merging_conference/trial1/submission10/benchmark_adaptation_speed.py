import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def kl_loss_fn(outputs, targets, T=2.0):
    p = nn.functional.log_softmax(outputs / T, dim=1)
    q = nn.functional.softmax(targets / T, dim=1)
    return nn.functional.kl_div(p, q, reduction='batchmean') * (T ** 2)

def adapt_head_vanilla(head, cal_feats, cal_targets, epochs=100, lr=1e-2):
    head = head.to(device)
    head.train()
    optimizer = optim.Adam(head.parameters(), lr=lr)
    dataset = TensorDataset(cal_feats, cal_targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        for feats, targets in loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = head(feats)
            loss = kl_loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
    return head

def adapt_head_sam(head, cal_feats, cal_targets, epochs=100, lr=1e-2, rho=0.05):
    head = head.to(device)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    dataset = TensorDataset(cal_feats, cal_targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        for feats, targets in loader:
            feats, targets = feats.to(device), targets.to(device)
            
            # Step 1: First forward-backward pass
            head.train()
            outputs = head(feats)
            loss = kl_loss_fn(outputs, targets)
            loss.backward()
            
            params = [p for p in head.parameters() if p.requires_grad]
            grads = [p.grad.clone() for p in params if p.grad is not None]
            grad_norm = torch.sqrt(sum([torch.sum(g ** 2) for g in grads]))
            
            if grad_norm > 0:
                scale = rho / grad_norm
                for p, g in zip(params, grads):
                    p.data.add_(g * scale)
                    
            optimizer.zero_grad()
            
            # Step 2: Second forward-backward pass
            outputs_perturbed = head(feats)
            loss_perturbed = kl_loss_fn(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            if grad_norm > 0:
                for p, g in zip(params, grads):
                    p.data.sub_(g * scale)
                    
            optimizer.step()
            optimizer.zero_grad()
            
    return head

def main():
    # Setup simulated data mimicking N_cal = 1000, 512-dim features, 10-class expert soft targets
    N_cal = 1000
    features_dim = 512
    num_classes = 10
    
    torch.manual_seed(42)
    cal_feats = torch.randn(N_cal, features_dim)
    cal_targets = torch.randn(N_cal, num_classes) # Soft targets
    
    # Define linear classification head
    head_init = nn.Linear(features_dim, num_classes)
    
    num_runs = 10
    vanilla_times = []
    sam_times = []
    
    print(f"Benchmarking {num_runs} runs on {device}...")
    
    # Warmup
    _ = adapt_head_vanilla(nn.Linear(features_dim, num_classes), cal_feats, cal_targets, epochs=5)
    _ = adapt_head_sam(nn.Linear(features_dim, num_classes), cal_feats, cal_targets, epochs=5)
    
    for run in range(num_runs):
        # Vanilla (SyMerge)
        head_v = nn.Linear(features_dim, num_classes)
        head_v.load_state_dict(head_init.state_dict())
        t0 = time.perf_counter()
        _ = adapt_head_vanilla(head_v, cal_feats, cal_targets, epochs=100)
        t1 = time.perf_counter()
        vanilla_times.append(t1 - t0)
        
        # SAM (SA-SyMerge)
        head_s = nn.Linear(features_dim, num_classes)
        head_s.load_state_dict(head_init.state_dict())
        t0 = time.perf_counter()
        _ = adapt_head_sam(head_s, cal_feats, cal_targets, epochs=100)
        t1 = time.perf_counter()
        sam_times.append(t1 - t0)
        
        print(f"Run {run + 1}/{num_runs}: Vanilla={vanilla_times[-1]:.4f}s | SAM={sam_times[-1]:.4f}s")
        
    vanilla_mean, vanilla_std = np.mean(vanilla_times), np.std(vanilla_times)
    sam_mean, sam_std = np.mean(sam_times), np.std(sam_times)
    
    print("\n--- BENCHMARK RESULTS ---")
    print(f"SyMerge Head Adaptation:       {vanilla_mean:.4f}s ± {vanilla_std:.4f}s")
    print(f"SA-SyMerge Head Adaptation:    {sam_mean:.4f}s ± {sam_std:.4f}s")
    print(f"Relative Overhead:             {(sam_mean / vanilla_mean - 1.0)*100:.1f}%")
    print(f"SA-SyMerge training speed is:   {1.0 / sam_mean:.2f} runs per second.")

if __name__ == "__main__":
    main()
