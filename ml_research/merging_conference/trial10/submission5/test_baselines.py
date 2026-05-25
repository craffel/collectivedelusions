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
        
    print("Evaluating with rho=0.03, damping_base=0.05:")
    
    # SAM-TTMM
    phase_accs_sam, overall_acc_sam, _, _, _ = evaluate_stream(
        expert_0, expert_1, stream_batches, regime="sam_ttmm",
        lr_base=150.0, rho=0.03, damping_base=0.05
    )
    print(f"SAM-TTMM: {overall_acc_sam*100:.4f}% (P1: {phase_accs_sam[0]*100:.2f}%, P2: {phase_accs_sam[1]*100:.2f}%, P3: {phase_accs_sam[2]*100:.2f}%, P4: {phase_accs_sam[3]*100:.2f}%, P5: {phase_accs_sam[4]*100:.2f}%)")
    
    # CG-MTTMM (Our best)
    phase_accs_cg, overall_acc_cg, _, _, _ = evaluate_stream(
        expert_0, expert_1, stream_batches, regime="cg_mttmm",
        lr_base=150.0, alpha=10.0, beta=50.0, rho=0.03, damping_base=0.05
    )
    print(f"CG-MTTMM: {overall_acc_cg*100:.4f}% (P1: {phase_accs_cg[0]*100:.2f}%, P2: {phase_accs_cg[1]*100:.2f}%, P3: {phase_accs_cg[2]*100:.2f}%, P4: {phase_accs_cg[3]*100:.2f}%, P5: {phase_accs_cg[4]*100:.2f}%)")

if __name__ == "__main__":
    main()
