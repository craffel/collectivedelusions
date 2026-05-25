import torch
import numpy as np
from run_ttmm import SimpleCNN, evaluate_stream, DataLoader, Subset, datasets, transforms

def main():
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
        
    _, sam_acc, _, _, _ = evaluate_stream(expert_0, expert_1, stream_batches, regime="sam_ttmm", lr_base=150.0)
    print(f"SAM-TTMM Accuracy: {sam_acc*100:.6f}%")
    
    _, cg_acc, _, _, _ = evaluate_stream(expert_0, expert_1, stream_batches, regime="cg_mttmm", lr_base=150.0, alpha=0.0, beta=0.0)
    print(f"CG-MTTMM (alpha=0, beta=0) Accuracy: {cg_acc*100:.6f}%")

if __name__ == "__main__":
    main()
