import torch
import numpy as np
import random
import time
from run_ttmm import SimpleCNN, evaluate_stream, DataLoader, Subset, datasets, transforms

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cpu")
    expert_0 = SimpleCNN().to(device)
    expert_1 = SimpleCNN().to(device)
    expert_0.load_state_dict(torch.load("models/expert_0.pt", map_location=device))
    expert_1.load_state_dict(torch.load("models/expert_1.pt", map_location=device))
    expert_0.eval()
    expert_1.eval()
    
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
        
    lr_candidates = [130.0, 140.0, 150.0, 160.0, 170.0]
    rhos = [0.02, 0.025, 0.03, 0.035, 0.04]
    dampings = [0.03, 0.04, 0.05, 0.06]
    alphas = [0.0, 5.0, 10.0, 50.0, 100.0]
    betas = [0.0, 10.0, 50.0, 100.0, 500.0]
    
    configs = []
    for lr in lr_candidates:
        for rho in rhos:
            for db in dampings:
                for alpha in alphas:
                    for beta in betas:
                        configs.append((lr, rho, db, alpha, beta))
                        
    # Randomly sample 100 configs to run sequentially (will take ~15-20s total)
    random.shuffle(configs)
    configs_to_run = configs[:100]
    
    print(f"Running {len(configs_to_run)} sequential evaluations...")
    best_acc = 0.0
    best_config = {}
    
    start_time = time.time()
    for i, (lr, rho, db, alpha, beta) in enumerate(configs_to_run):
        try:
            torch.manual_seed(42)
            np.random.seed(42)
            phase_accs, overall_acc, _, _, _ = evaluate_stream(
                expert_0, expert_1, stream_batches, regime="cg_mttmm",
                lr_base=lr, alpha=alpha, beta=beta, rho=rho, damping_base=db
            )
            print(f"[{i+1}/100] lr={lr:<5} | rho={rho:<5} | db={db:<5} | alpha={alpha:<5} | beta={beta:<5} -> Acc: {overall_acc*100:.4f}%")
            if overall_acc > best_acc:
                best_acc = overall_acc
                best_config = {
                    "lr_base": lr, "rho": rho, "damping_base": db,
                    "alpha": alpha, "beta": beta, "phase_accs": phase_accs
                }
        except Exception as e:
            print(f"Error on config {i}: {e}")
            
    print("\n" + "="*50)
    print("BEST CONFIG FOUND:")
    print(best_config)
    print(f"Best Accuracy: {best_acc*100:.4f}%")
    print("="*50)

if __name__ == "__main__":
    main()
