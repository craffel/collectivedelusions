import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
import multiprocessing
from run_ttmm import SimpleCNN, evaluate_stream

# Global variables for workers
expert_0 = None
expert_1 = None
stream_batches = None

def init_worker():
    global expert_0, expert_1, stream_batches
    device = torch.device("cpu")
    
    # Load experts
    expert_0 = SimpleCNN().to(device)
    expert_1 = SimpleCNN().to(device)
    expert_0.load_state_dict(torch.load("models/expert_0.pt", map_location=device))
    expert_1.load_state_dict(torch.load("models/expert_1.pt", map_location=device))
    expert_0.eval()
    expert_1.eval()
    
    # Load stream batches
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_test = datasets.MNIST(root="data", train=False, download=False, transform=transform)
    fmnist_test = datasets.FashionMNIST(root="data", train=False, download=False, transform=transform)
    kmnist_test = datasets.KMNIST(root="data", train=False, download=False, transform=transform)
    
    mnist_clean_loader = DataLoader(Subset(mnist_test, list(range(640))), batch_size=64, shuffle=False)
    fmnist_clean_loader = DataLoader(Subset(fmnist_test, list(range(640))), batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(Subset(kmnist_test, list(range(640))), batch_size=64, shuffle=False)
    
    stream_batches = []
    for x, y in mnist_clean_loader:
        stream_batches.append((x, y))
    for x, y in mnist_clean_loader:
        noisy_x = x + 0.6 * torch.randn_like(x)
        noisy_x = torch.clamp(noisy_x, -1.0, 1.0)
        stream_batches.append((noisy_x, y))
    for x, y in fmnist_clean_loader:
        stream_batches.append((x, y))
    for x, y in fmnist_clean_loader:
        noisy_x = x + 0.6 * torch.randn_like(x)
        noisy_x = torch.clamp(noisy_x, -1.0, 1.0)
        stream_batches.append((noisy_x, y))
    for x, y in kmnist_loader:
        stream_batches.append((x, y))

def run_one_config(args):
    lr, alpha, beta, rho, db = args
    try:
        # Re-set random seeds per evaluation to ensure determinism
        torch.manual_seed(42)
        np.random.seed(42)
        
        phase_accs, overall_acc, _, _, _ = evaluate_stream(
            expert_0, expert_1, stream_batches, regime="cg_mttmm",
            lr_base=lr, alpha=alpha, beta=beta, rho=rho, damping_base=db
        )
        return (lr, alpha, beta, rho, db, overall_acc, phase_accs)
    except Exception as e:
        print(f"Error running config {args}: {e}")
        return (lr, alpha, beta, rho, db, 0.0, [])

def main():
    print("Preparing CPU Multiprocessing Sweep with Active Progress Reporting...")
    
    # Grid choices
    lr_candidates = [140.0, 150.0, 160.0, 170.0, 180.0, 200.0]
    alphas = [0.0, 100.0, 200.0, 500.0, 800.0, 1200.0]
    betas = [0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0]
    rhos = [0.025, 0.030, 0.035, 0.040]
    dampings = [0.03, 0.04, 0.05, 0.06]
    
    configs_to_run = []
    for lr in lr_candidates:
        for alpha in alphas:
            for beta in betas:
                for rho in rhos:
                    for db in dampings:
                        configs_to_run.append((lr, alpha, beta, rho, db))
                        
    # Shuffle and sample 250 configurations to run extremely fast!
    import random
    random.seed(42)
    random.shuffle(configs_to_run)
    configs_to_run = configs_to_run[:250]
    
    total_configs = len(configs_to_run)
    print(f"Total configurations to evaluate (sampled): {total_configs}")
    
    num_workers = min(multiprocessing.cpu_count(), 4)
    print(f"Starting parallel sweep with {num_workers} worker processes...")
    
    start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
    
    pool = multiprocessing.Pool(processes=num_workers, initializer=init_worker)
    
    best_acc = 0.0
    best_config = None
    completed = 0
    
    # Use imap_unordered for streaming results with real-time feedback
    for result in pool.imap_unordered(run_one_config, configs_to_run):
        lr, alpha, beta, rho, db, acc, pa = result
        completed += 1
        
        if acc > best_acc:
            best_acc = acc
            best_config = (lr, alpha, beta, rho, db, pa)
            print(f"[{completed}/{total_configs}] NEW GLOBAL BEST: lr={lr:<5} | alpha={alpha:<5} | beta={beta:<5} | rho={rho:<5} | db={db:<5} -> Acc: {acc*100:.4f}%")
        elif completed % 20 == 0:
            elapsed = time.clock_gettime(time.CLOCK_MONOTONIC) - start_time
            print(f"[{completed}/{total_configs}] Completed {completed} configs in {elapsed:.1f}s (Current Best: {best_acc*100:.4f}%)")
            
    pool.close()
    pool.join()
    
    elapsed = time.clock_gettime(time.CLOCK_MONOTONIC) - start_time
    print(f"\nSweep fully completed in {elapsed:.2f} seconds.")
    print("==================================================")
    print("GLOBAL BEST CONFIG FOUND:")
    lr, alpha, beta, rho, db, pa = best_config
    print(f"  lr_base:      {lr}")
    print(f"  alpha:        {alpha}")
    print(f"  beta:         {beta}")
    print(f"  rho:          {rho}")
    print(f"  damping_base: {db}")
    print(f"  Overall Accuracy: {best_acc*100:.4f}%")
    print(f"  Phase Accuracies: {[round(a*100, 2) for a in pa]}")
    print("==================================================")

if __name__ == "__main__":
    main()
