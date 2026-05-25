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
    expert_0.load_state_dict(torch.load("models/expert_0.pt", map_location=device, weights_only=True))
    expert_1.load_state_dict(torch.load("models/expert_1.pt", map_location=device, weights_only=True))
    
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
        
    alphas = [0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    betas = [0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    print("Sweeping alpha and beta for rho=0.03, damping_base=0.05, lr_base=150.0:")
    best_acc = 0.0
    best_cfg = None
    for alpha in alphas:
        for beta in betas:
            phase_accs, overall_acc, _, _, _ = evaluate_stream(
                expert_0, expert_1, stream_batches, regime="cg_mttmm",
                lr_base=150.0, alpha=alpha, beta=beta, rho=0.03, damping_base=0.05
            )
            print(f"alpha={alpha:<5} | beta={beta:<5} -> Acc: {overall_acc*100:.4f}% (P1: {phase_accs[0]*100:.1f}%, P2: {phase_accs[1]*100:.1f}%, P3: {phase_accs[2]*100:.1f}%, P4: {phase_accs[3]*100:.1f}%, P5: {phase_accs[4]*100:.1f}%)")
            if overall_acc > best_acc:
                best_acc = overall_acc
                best_cfg = (alpha, beta, phase_accs)
                
    print(f"\nBest Config: alpha={best_cfg[0]}, beta={best_cfg[1]} with Acc: {best_acc*100:.4f}%")

if __name__ == "__main__":
    main()
