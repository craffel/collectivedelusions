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
        
    lr_candidates = [100.0, 150.0]
    alphas = [0.0, 100.0, 500.0, 1000.0]
    betas = [0.0, 1000.0, 5000.0, 10000.0]
    
    for lr in lr_candidates:
        for alpha in alphas:
            for beta in betas:
                _, overall_acc, _, _, _ = evaluate_stream(
                    expert_0, expert_1, stream_batches, regime="cg_mttmm",
                    lr_base=lr, alpha=alpha, beta=beta
                )
                print(f"lr={lr:<5} | alpha={alpha:<6} | beta={beta:<7} -> Acc: {overall_acc*100:.4f}%")

if __name__ == "__main__":
    main()
