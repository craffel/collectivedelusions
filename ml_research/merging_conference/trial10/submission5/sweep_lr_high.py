import torch
import numpy as np
from run_ttmm import SimpleCNN, evaluate_stream, DataLoader, Subset, datasets, transforms

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cpu")
    print("Loading experts...")
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
        
    print(f"Stream loaded. Total batches: {len(stream_batches)}")
    
    lr_candidates = [40.0, 60.0, 80.0, 100.0, 120.0, 150.0]
    regimes = ["sam_ttmm", "cg_mttmm"]
    
    for regime in regimes:
        print(f"\nSweeping high learning rates for regime: {regime.upper()}")
        for lr in lr_candidates:
            if regime == "cg_mttmm":
                phase_accs, overall_acc, _, _, _ = evaluate_stream(expert_0, expert_1, stream_batches, regime=regime, lr_base=lr, alpha=1000.0, beta=7500.0)
            else:
                phase_accs, overall_acc, _, _, _ = evaluate_stream(expert_0, expert_1, stream_batches, regime=regime, lr_base=lr)
            print(f"  lr_base={lr:<5} -> Overall Accuracy: {overall_acc*100:.2f}% (P1: {phase_accs[0]*100:.1f}%, P2: {phase_accs[1]*100:.1f}%, P3: {phase_accs[2]*100:.1f}%, P4: {phase_accs[3]*100:.1f}%, P5: {phase_accs[4]*100:.1f}%)")

if __name__ == "__main__":
    main()
