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
    
    # Sweep 1: Learning Rate lr_lambda for PC-Merge + OPR on Sequential Clean & Sequential Gaussian Noise
    lr_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    print("\n=== SWEEPING LEARNING RATE lr_lambda ===")
    print("Method: PC-Merge with OPR (Ours)")
    print("lr_lambda | Clean Acc | Gaussian Noise Acc")
    print("----------|-----------|-------------------")
    for lr in lr_values:
        acc_clean = evaluate_method("PC-Merge with OPR (Ours)", "Clean", "Sequential", expert_data, fim_priors, device, lr_lambda_custom=lr)
        acc_noise = evaluate_method("PC-Merge with OPR (Ours)", "Gaussian Noise", "Sequential", expert_data, fim_priors, device, lr_lambda_custom=lr)
        print(f"{lr:<9} | {acc_clean*100:6.2f}%  | {acc_noise*100:16.2f}%")
        
    # Sweep 2: OPR Threshold alpha on Sequential Clean & Sequential Gaussian Noise
    alpha_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    print("\n=== SWEEPING OPR THRESHOLD alpha ===")
    print("Method: PC-Merge with OPR (Ours)")
    print("alpha | Clean Acc | Gaussian Noise Acc")
    print("------|-----------|-------------------")
    for alpha in alpha_values:
        acc_clean = evaluate_method("PC-Merge with OPR (Ours)", "Clean", "Sequential", expert_data, fim_priors, device, threshold_multiplier_custom=alpha)
        acc_noise = evaluate_method("PC-Merge with OPR (Ours)", "Gaussian Noise", "Sequential", expert_data, fim_priors, device, threshold_multiplier_custom=alpha)
        print(f"{alpha:<5} | {acc_clean*100:6.2f}%  | {acc_noise*100:16.2f}%")

if __name__ == "__main__":
    main()
