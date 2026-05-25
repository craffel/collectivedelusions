import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import eval_tta
from eval_tta import evaluate_method, compute_fim_priors

def run_noise_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Precompute FIM priors
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
    
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
    methods = ["Standard TTA", "PC-Merge with OPR (Ours)"]
    
    results = {m: {} for m in methods}
    
    for s in sigmas:
        print(f"\n--- Evaluating Noise Sigma: {s} ---")
        
        # Override the apply_gaussian_noise function in eval_tta dynamically
        def custom_gaussian_noise(x, sigma=s):
            return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)
            
        eval_tta.apply_gaussian_noise = custom_gaussian_noise
        
        for m in methods:
            print(f"Evaluating {m} on Gaussian Noise (Sigma = {s})...")
            acc = evaluate_method(m, "Gaussian Noise", "Sequential", expert_data, fim_priors, device)
            results[m][s] = acc
            print(f"Accuracy: {acc*100:.2f}%")
            
    # Print results summary
    print("\n### NOISE SEVERITY SWEEP RESULTS")
    print("| Method | Sigma=0.1 | Sigma=0.2 | Sigma=0.3 | Sigma=0.4 | Sigma=0.5 |")
    print("|--------|-----------|-----------|-----------|-----------|-----------|")
    for m in methods:
        row = [f"{results[m][s]*100:.2f}%" for s in sigmas]
        print(f"| {m} | " + " | ".join(row) + " |")

if __name__ == "__main__":
    run_noise_sweep()
