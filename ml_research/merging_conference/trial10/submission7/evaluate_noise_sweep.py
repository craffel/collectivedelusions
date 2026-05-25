import torch
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import sys

# Import functions from original evaluate_ttmm
from evaluate_ttmm import SimpleCNN, evaluate_method

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_test_batches(seed, mnist_test, fmnist_test, kmnist_test, sigma):
    # Set seed before constructing noise to ensure reproducibility
    set_seed(seed)
    test_batches = []
    
    # Phase 1: Clean MNIST (batches 0-9)
    loader_mnist_clean = DataLoader(Subset(mnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_mnist_clean:
        test_batches.append((x, y))
        
    # Phase 2: Noisy MNIST (batches 10-19)
    loader_mnist_noisy = DataLoader(Subset(mnist_test, list(range(640, 1280))), batch_size=64, shuffle=False)
    for x, y in loader_mnist_noisy:
        noise = torch.randn_like(x) * sigma
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_batches.append((x_noisy, y))
        
    # Phase 3: Clean FashionMNIST (batches 20-29)
    loader_fashion_clean = DataLoader(Subset(fmnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_fashion_clean:
        test_batches.append((x, y))
        
    # Phase 4: Noisy FashionMNIST (batches 30-39)
    loader_fashion_noisy = DataLoader(Subset(fmnist_test, list(range(640, 1280))), batch_size=64, shuffle=False)
    for x, y in loader_fashion_noisy:
        noise = torch.randn_like(x) * sigma
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_batches.append((x_noisy, y))
        
    # Phase 5: Novel KMNIST (batches 40-49)
    loader_kmnist = DataLoader(Subset(kmnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_kmnist:
        test_batches.append((x, y))
        
    return test_batches

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating noise sweep on device: {device}")
    
    # Load expert models
    model_mnist = SimpleCNN().to(device)
    model_mnist.load_state_dict(torch.load("expert_mnist.pth", map_location=device))
    model_mnist.eval()
    
    model_fashion = SimpleCNN().to(device)
    model_fashion.load_state_dict(torch.load("expert_fashion.pth", map_location=device))
    model_fashion.eval()
    
    # Load prototypes
    protos = torch.load("prototypes.pth", map_location=device)
    
    # Prepare datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform)
    
    seeds = [42, 43, 44, 45, 46]
    noise_scales = [0.0, 0.2, 0.4, 0.6, 0.8]
    methods = [
        "Method A (Fixed TTA + Reset)",
        "Method B (CL W-Fisher + SCTS L2)",
        "Method C (CL W-Fisher + A-SCTS)",
        "Method D (CP-AM)",
        "Method E (Original BK-AHR, Scale-Mismatched)",
        "Method F (BK-AHR with Normalized L2)",
        "Method G (CSAIR, Ours)",
        "Method H (CSASAM, Ours)"
    ]
    
    # Initialize results structures: Method -> noise_scale -> list of overalls (across seeds)
    results = {m: {sigma: [] for sigma in noise_scales} for m in methods}
    
    for sigma in noise_scales:
        print("\n" + "="*60)
        print(f"RUNNING NOISE SWEEP WITH SIGMA = {sigma}")
        print("="*60)
        
        for seed in seeds:
            print(f"--- Seed {seed} ---")
            test_batches = get_test_batches(seed, mnist_test, fmnist_test, kmnist_test, sigma)
            
            for m in methods:
                set_seed(seed)
                _, overall = evaluate_method(m, test_batches, model_mnist, model_fashion, protos, device)
                results[m][sigma].append(overall)
                
    # Calculate statistics
    print("\n\n" + "="*80)
    print("NOISE SWEEP STATISTICAL SUMMARY (5 Seeds)")
    print("="*80)
    
    # Print markdown-style table
    header = "| Method | " + " | ".join([f"sigma = {sigma}" for sigma in noise_scales]) + " |"
    divider = "|---| " + " | ".join(["---" for _ in noise_scales]) + " |"
    print(header)
    print(divider)
    
    for m in methods:
        row_str = f"| {m} |"
        for sigma in noise_scales:
            overalls = np.array(results[m][sigma])
            mean = np.mean(overalls)
            std = np.std(overalls)
            row_str += f" {mean:.2f}±{std:.2f}% |"
        print(row_str)
        
    # Print LaTeX-style table code
    print("\n\n" + "="*80)
    print("LATEX TABLE ROWS FOR NOISE SWEEP")
    print("="*80)
    for m in methods:
        m_clean = m
        if "Ours" in m:
            m_clean = f"\\textbf{{{m}}}"
        row_str = f"{m_clean} & "
        cells = []
        for sigma in noise_scales:
            overalls = np.array(results[m][sigma])
            mean = np.mean(overalls)
            std = np.std(overalls)
            cells.append(f"{mean:.2f}\\% $\\pm$ {std:.2f}\\%")
        row_str += " & ".join(cells) + " \\\\"
        print(row_str)

if __name__ == "__main__":
    main()