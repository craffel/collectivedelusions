import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from eval_tta import evaluate_method, compute_fim_priors

def run_batch_sweep():
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
    
    batch_sizes = [16, 32, 64, 128]
    methods = ["Standard TTA", "PC-Merge with OPR (Ours)"]
    domains = ["Clean", "Gaussian Noise"]
    
    results = {m: {d: {} for d in domains} for m in methods}
    
    for bs in batch_sizes:
        print(f"\n--- Evaluating Batch Size: {bs} ---")
        mnist_loader = DataLoader(mnist_test, batch_size=bs, shuffle=False)
        fashion_loader = DataLoader(fashion_test, batch_size=bs, shuffle=False)
        kmnist_loader = DataLoader(kmnist_test, batch_size=bs, shuffle=False)
        expert_data = (mnist_loader, fashion_loader, kmnist_loader)
        
        for d in domains:
            for m in methods:
                print(f"Evaluating {m} on {d} (Batch Size = {bs})...")
                acc = evaluate_method(m, d, "Sequential", expert_data, fim_priors, device)
                results[m][d][bs] = acc
                print(f"Accuracy: {acc*100:.2f}%")
                
    # Print results summary
    print("\n### BATCH SIZE SWEEP RESULTS")
    for d in domains:
        print(f"\n#### Domain: {d}")
        print("| Method | B=16 | B=32 | B=64 | B=128 |")
        print("|--------|------|------|------|-------|")
        for m in methods:
            row = [f"{results[m][d][bs]*100:.2f}%" for bs in batch_sizes]
            print(f"| {m} | " + " | ".join(row) + " |")

if __name__ == "__main__":
    run_batch_sweep()
