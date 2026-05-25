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

def get_test_batches(seed, mnist_test, fmnist_test, kmnist_test):
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
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_batches.append((x_noisy, y))
        
    # Phase 3: Clean FashionMNIST (batches 20-29)
    loader_fashion_clean = DataLoader(Subset(fmnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_fashion_clean:
        test_batches.append((x, y))
        
    # Phase 4: Noisy FashionMNIST (batches 30-39)
    loader_fashion_noisy = DataLoader(Subset(fmnist_test, list(range(640, 1280))), batch_size=64, shuffle=False)
    for x, y in loader_fashion_noisy:
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_batches.append((x_noisy, y))
        
    # Phase 5: Novel KMNIST (batches 40-49)
    loader_kmnist = DataLoader(Subset(kmnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_kmnist:
        test_batches.append((x, y))
        
    return test_batches

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating multi-seed on device: {device}")
    
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
    
    # Initialize results structures
    # Method -> list of (list of 5 phase accuracies, overall accuracy)
    results_by_method = {m: [] for m in methods}
    
    for seed in seeds:
        print("\n" + "="*60)
        print(f"RUNNING WITH SEED {seed}")
        print("="*60)
        
        test_batches = get_test_batches(seed, mnist_test, fmnist_test, kmnist_test)
        
        for m in methods:
            # We set seed again before each method to ensure any internal torch randomness is consistent
            set_seed(seed)
            accs, overall = evaluate_method(m, test_batches, model_mnist, model_fashion, protos, device)
            results_by_method[m].append((accs, overall))
            
    # Calculate statistics
    print("\n\n" + "="*80)
    print("MULTI-SEED STATISTICAL SUMMARY (5 Seeds)")
    print("="*80)
    
    # Print markdown-style table
    print("| Method | C-MN | N-MN | C-FN | N-FN | Nov-K | Overall |")
    print("|---|---|---|---|---|---|---|")
    
    for m in methods:
        runs = results_by_method[m]
        # runs is list of length 5, each element is (accs_list, overall_acc)
        # accs_list has 5 elements
        all_accs = np.array([r[0] for r in runs]) # (5, 5)
        all_overalls = np.array([r[1] for r in runs]) # (5,)
        
        means = np.mean(all_accs, axis=0)
        stds = np.std(all_accs, axis=0)
        
        mean_overall = np.mean(all_overalls)
        std_overall = np.std(all_overalls)
        
        print(f"| {m} "
              f"| {means[0]:.2f}±{stds[0]:.2f}% "
              f"| {means[1]:.2f}±{stds[1]:.2f}% "
              f"| {means[2]:.2f}±{stds[2]:.2f}% "
              f"| {means[3]:.2f}±{stds[3]:.2f}% "
              f"| {means[4]:.2f}±{stds[4]:.2f}% "
              f"| {mean_overall:.2f}±{std_overall:.2f}% |")
              
    # Print LaTeX-style table row code
    print("\n\n" + "="*80)
    print("LATEX TABLE ROWS FOR paper.tex")
    print("="*80)
    for m in methods:
        runs = results_by_method[m]
        all_accs = np.array([r[0] for r in runs]) # (5, 5)
        all_overalls = np.array([r[1] for r in runs]) # (5,)
        
        means = np.mean(all_accs, axis=0)
        stds = np.std(all_accs, axis=0)
        mean_overall = np.mean(all_overalls)
        std_overall = np.std(all_overalls)
        
        # Replace method name prefix for LaTeX formatting if desired
        m_clean = m
        if "Ours" in m:
            m_clean = f"\\textbf{{{m}}}"
            
        print(f"{m_clean} & "
              f"{means[0]:.2f}\\% $\\pm$ {stds[0]:.2f}\\% & "
              f"{means[1]:.2f}\\% $\\pm$ {stds[1]:.2f}\\% & "
              f"{means[2]:.2f}\\% $\\pm$ {stds[2]:.2f}\\% & "
              f"{means[3]:.2f}\\% $\\pm$ {stds[3]:.2f}\\% & "
              f"{means[4]:.2f}\\% $\\pm$ {stds[4]:.2f}\\% & "
              f"{mean_overall:.2f}\\% $\\pm$ {std_overall:.2f}\\% \\\\")

if __name__ == "__main__":
    main()
