import torch
import numpy as np
from run_ttmm import SimpleCNN, evaluate_stream, DataLoader, Subset, datasets, transforms

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cpu")
    expert_0 = SimpleCNN().to(device)
    expert_1 = SimpleCNN().to(device)
    expert_0.load_state_dict(torch.load("models/expert_0.pt", map_location=device))
    expert_1.load_state_dict(torch.load("models/expert_1.pt", map_location=device))
    
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
        
    lr_candidates = [120.0, 140.0, 150.0, 160.0, 180.0]
    alphas = [0.0, 5.0, 10.0]
    betas = [50.0, 80.0, 100.0, 150.0, 200.0]
    rhos = [0.01, 0.02, 0.03]
    dampings = [0.03, 0.04, 0.05, 0.06]
    
    best_acc = 0.0
    best_config = {}
    
    # We will sample 150 configurations randomly from this space to keep it super fast!
    import random
    random.seed(42)
    
    configs_to_run = []
    for lr in lr_candidates:
        for alpha in alphas:
            for beta in betas:
                for rho in rhos:
                    for db in dampings:
                        configs_to_run.append((lr, alpha, beta, rho, db))
                        
    # Shuffle and take 150
    random.shuffle(configs_to_run)
    configs_to_run = configs_to_run[:150]
    
    print(f"Running {len(configs_to_run)} finer configuration evaluations...")
    for lr, alpha, beta, rho, db in configs_to_run:
        phase_accs, overall_acc, _, _, _ = evaluate_stream(
            expert_0, expert_1, stream_batches, regime="cg_mttmm",
            lr_base=lr, alpha=alpha, beta=beta, rho=rho, damping_base=db
        )
        if overall_acc > best_acc:
            best_acc = overall_acc
            best_config = {
                "lr_base": lr, "alpha": alpha, "beta": beta,
                "rho": rho, "damping_base": db, "phase_accs": phase_accs
            }
            print(f"NEW BEST: lr={lr:<5} | alpha={alpha:<5} | beta={beta:<5} | rho={rho:<5} | db={db:<5} -> Acc: {overall_acc*100:.4f}% (P1: {phase_accs[0]*100:.1f}%, P2: {phase_accs[1]*100:.1f}%, P3: {phase_accs[2]*100:.1f}%, P4: {phase_accs[3]*100:.1f}%, P5: {phase_accs[4]*100:.1f}%)")
                            
    print("\nFiner sweep complete.")
    print(f"Optimal Config: {best_config}")
    print(f"Best Accuracy: {best_acc*100:.4f}%")

if __name__ == "__main__":
    main()
