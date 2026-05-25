import torch
import numpy as np
from run_ttmm import SimpleCNN, evaluate_stream, DataLoader, Subset, datasets, transforms

def main():
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
        
    _, _, curvature, _, _ = evaluate_stream(expert_0, expert_1, stream_batches, regime="cg_mttmm", lr_base=10.0)
    
    # curvature has 50 elements (one per batch)
    # Print statistics per phase
    phases = [
        "Phase 1: Clean MNIST",
        "Phase 2: Noisy MNIST",
        "Phase 3: Clean FashionMNIST",
        "Phase 4: Noisy FashionMNIST",
        "Phase 5: Novel KMNIST"
    ]
    
    for i, phase_name in enumerate(phases):
        phase_curv = curvature[i*10 : (i+1)*10]
        mean_val = np.mean(phase_curv)
        std_val = np.std(phase_curv)
        max_val = np.max(phase_curv)
        print(f"{phase_name:<30} | Mean: {mean_val:.6f} | Std: {std_val:.6f} | Max: {max_val:.6f}")

if __name__ == "__main__":
    main()
