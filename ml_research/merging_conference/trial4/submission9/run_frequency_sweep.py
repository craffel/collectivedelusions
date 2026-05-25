import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from eval_tta import evaluate_method, compute_fim_priors

def run_frequency_sweep():
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
    
    block_sizes = [1, 2, 5, 10, 25, 50]
    methods = ["Static Merged", "Standard TTA", "PC-Merge with OPR (Ours)"]
    domains = ["Clean", "Gaussian Noise"]
    
    results = {m: {d: {} for d in domains} for m in methods}
    
    for bs in block_sizes:
        print(f"\n--- Evaluating Block Size (Task Switch Interval): {bs} ---")
        for d in domains:
            for m in methods:
                print(f"Evaluating {m} on {d} (Block Size = {bs})...")
                acc = evaluate_method(m, d, "Sequential", expert_data, fim_priors, device, block_size=bs)
                results[m][d][bs] = acc
                print(f"Accuracy: {acc*100:.2f}%")
                
    # Print results summary in Markdown format
    print("\n### TASK TRANSITION FREQUENCY SWEEP RESULTS")
    for d in domains:
        print(f"\n#### Domain: {d}")
        print("| Method | N=1 (Alternating) | N=2 | N=5 | N=10 | N=25 | N=50 (Sequential) |")
        print("|--------|-------------------|-----|-----|------|------|-------------------|")
        for m in methods:
            row = [f"{results[m][d][bs]*100:.2f}%" for bs in block_sizes]
            print(f"| {m} | " + " | ".join(row) + " |")

if __name__ == "__main__":
    run_frequency_sweep()
