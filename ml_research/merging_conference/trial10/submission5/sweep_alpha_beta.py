import torch
import numpy as np
from run_ttmm import SimpleCNN, evaluate_stream, DataLoader, Subset, datasets, transforms

def main():
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
    
    # Run sam_ttmm baseline at lr_base=40.0
    _, sam_acc, _, _, _ = evaluate_stream(expert_0, expert_1, stream_batches, regime="sam_ttmm", lr_base=40.0)
    print(f"SAM-TTMM Baseline Accuracy (lr_base=40.0): {sam_acc*100:.2f}%")
    
    alphas = [0.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0]
    betas = [0.0, 1000.0, 3000.0, 5000.0, 7500.0, 10000.0]
    
    best_acc = 0.0
    best_alpha = 0.0
    best_beta = 0.0
    
    print("\nGrid Search over Alpha and Beta for CG_MTTMM (lr_base=40.0):")
    for alpha in alphas:
        for beta in betas:
            phase_accs, overall_acc, _, _, _ = evaluate_stream(
                expert_0, expert_1, stream_batches, regime="cg_mttmm",
                lr_base=40.0, alpha=alpha, beta=beta
            )
            print(f"  alpha={alpha:<6} | beta={beta:<7} -> Accuracy: {overall_acc*100:.2f}% (P1: {phase_accs[0]*100:.1f}%, P2: {phase_accs[1]*100:.1f}%, P3: {phase_accs[2]*100:.1f}%, P4: {phase_accs[3]*100:.1f}%, P5: {phase_accs[4]*100:.1f}%)")
            if overall_acc > best_acc:
                best_acc = overall_acc
                best_alpha = alpha
                best_beta = beta
                
    print(f"\nBest CG_MTTMM config: alpha={best_alpha}, beta={best_beta} with Overall Accuracy of {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
