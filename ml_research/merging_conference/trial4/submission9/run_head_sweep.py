import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from eval_tta import evaluate_method, compute_fim_priors

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Compute FIM priors once
    fim_priors = compute_fim_priors(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = datasets.MNIST(root="./data", train=False, transform=transform)
    fashion_test = datasets.FashionMNIST(root="./data", train=False, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    fashion_loader = DataLoader(fashion_test, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    expert_data = (mnist_loader, fashion_loader, kmnist_loader)
    
    # Sweep: Learning Rate lr_head for PC-Merge + OPR on Sequential Clean & Sequential Gaussian Noise
    lr_head_values = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    print("\n=== SWEEPING CLASSIFICATION HEAD LEARNING RATE lr_head ===")
    print("Method: PC-Merge with OPR (Ours)")
    print("lr_head   | Clean Acc | Gaussian Noise Acc")
    print("----------|-----------|-------------------")
    for lr_head in lr_head_values:
        acc_clean = evaluate_method("PC-Merge with OPR (Ours)", "Clean", "Sequential", expert_data, fim_priors, device, lr_head_custom=lr_head)
        acc_noise = evaluate_method("PC-Merge with OPR (Ours)", "Gaussian Noise", "Sequential", expert_data, fim_priors, device, lr_head_custom=lr_head)
        print(f"{lr_head:<9.5f} | {acc_clean*100:6.2f}%  | {acc_noise*100:16.2f}%")

if __name__ == "__main__":
    main()
