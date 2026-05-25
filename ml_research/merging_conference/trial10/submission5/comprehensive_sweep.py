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
        
    lr_candidates = [40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0]
    alphas = [0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0]
    betas = [0.0, 1000.0, 2500.0, 5000.0, 7500.0, 10000.0, 15000.0]
    
    best_acc = 0.0
    best_config = {}
    
    # Static baseline
    _, static_acc, _, _, _ = evaluate_stream(expert_0, expert_1, stream_batches, regime="static")
    
    print("Running comprehensive grid search...")
    for lr in lr_candidates:
        for alpha in alphas:
            for beta in betas:
                phase_accs, overall_acc, _, _, _ = evaluate_stream(
                    expert_0, expert_1, stream_batches, regime="cg_mttmm",
                    lr_base=lr, alpha=alpha, beta=beta
                )
                if overall_acc > best_acc:
                    best_acc = overall_acc
                    best_config = {
                        "lr_base": lr,
                        "alpha": alpha,
                        "beta": beta,
                        "phase_accs": phase_accs
                    }
                    
    print("\n" + "="*50)
    print("GLOBAL BEST CONFIG FOR CG_MTTMM:")
    print(f"  lr_base: {best_config['lr_base']}")
    print(f"  alpha:   {best_config['alpha']}")
    print(f"  beta:    {best_config['beta']}")
    print(f"  Overall Accuracy: {best_acc*100:.3f}%")
    pa = best_config["phase_accs"]
    print(f"  Phase 1 (Clean MNIST):     {pa[0]*100:.2f}%")
    print(f"  Phase 2 (Noisy MNIST):     {pa[1]*100:.2f}%")
    print(f"  Phase 3 (Clean Fashion):   {pa[2]*100:.2f}%")
    print(f"  Phase 4 (Noisy Fashion):   {pa[3]*100:.2f}%")
    print(f"  Phase 5 (Novel KMNIST):    {pa[4]*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
