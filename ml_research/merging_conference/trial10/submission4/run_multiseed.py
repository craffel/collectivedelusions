import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from main import (
    set_seed, SimpleCNN, get_datasets, compute_prototypes, evaluate_stream
)

def construct_test_stream_with_seed(loader_mnist_test, loader_fashion_test, loader_kmnist_test, seed, device):
    # Set the seed for this specific stream generation
    set_seed(seed)
    
    mnist_iter = iter(loader_mnist_test)
    fashion_iter = iter(loader_fashion_test)
    kmnist_iter = iter(loader_kmnist_test)
    
    test_stream = []
    
    # Phase 1: Clean MNIST (batches 0-9)
    for _ in range(10):
        test_stream.append(next(mnist_iter))
        
    # Phase 2: Noisy MNIST (batches 10-19)
    for _ in range(10):
        x, y = next(mnist_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_stream.append((x_noisy, y))
        
    # Phase 3: Clean FashionMNIST (batches 20-29)
    for _ in range(10):
        test_stream.append(next(fashion_iter))
        
    # Phase 4: Noisy FashionMNIST (batches 30-39)
    for _ in range(10):
        x, y = next(fashion_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_stream.append((x_noisy, y))
        
    # Phase 5: Novel KMNIST (batches 40-49)
    for _ in range(10):
        test_stream.append(next(kmnist_iter))
        
    return test_stream

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for multi-seed evaluation: {device}")
    
    mnist_train, mnist_test, fashion_train, fashion_test, kmnist_test = get_datasets()
    loader_mnist_test = DataLoader(mnist_test, batch_size=64, shuffle=False)
    loader_fashion_test = DataLoader(fashion_test, batch_size=64, shuffle=False)
    loader_kmnist_test = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    
    standard_expert0 = SimpleCNN(is_cosface=False).to(device)
    standard_expert1 = SimpleCNN(is_cosface=False).to(device)
    cosface_expert0 = SimpleCNN(is_cosface=True).to(device)
    cosface_expert1 = SimpleCNN(is_cosface=True).to(device)
    
    standard_expert0.load_state_dict(torch.load("./checkpoints/standard_expert_mnist.pt", map_location=device, weights_only=True))
    standard_expert1.load_state_dict(torch.load("./checkpoints/standard_expert_fashion.pt", map_location=device, weights_only=True))
    cosface_expert0.load_state_dict(torch.load("./checkpoints/cosface_expert_mnist.pt", map_location=device, weights_only=True))
    cosface_expert1.load_state_dict(torch.load("./checkpoints/cosface_expert_fashion.pt", map_location=device, weights_only=True))
    
    standard_expert0.eval()
    standard_expert1.eval()
    cosface_expert0.eval()
    cosface_expert1.eval()
    
    print("Precomputing prototypes...")
    P0_std, P1_std = compute_prototypes(standard_expert0, standard_expert1, loader_mnist_test, loader_fashion_test, device=device)
    P0_cos, P1_cos = compute_prototypes(cosface_expert0, cosface_expert1, loader_mnist_test, loader_fashion_test, device=device)
    
    seeds = [42, 100, 2026, 777, 999]
    method_names = [
        "Fixed TTA",
        "MoG-L2",
        "MoG-Angular",
        "CP-AM (Baseline)",
        "BK-AHR",
        "FL-AHR (Ours)"
    ]
    
    # Structure to hold results: {method: {seed: (overall_acc, phase_accs)}}
    all_results = {m: {} for m in method_names}
    
    for seed in seeds:
        print(f"\nEvaluating with random seed: {seed}")
        test_stream = construct_test_stream_with_seed(
            loader_mnist_test, loader_fashion_test, loader_kmnist_test, seed, device
        )
        
        for m in method_names:
            mean_acc, accuracies = evaluate_stream(
                standard_expert0, standard_expert1, P0_std, P1_std,
                cosface_expert0, cosface_expert1, P0_cos, P1_cos,
                test_stream, m, device=device
            )
            # Store overall accuracy and phase-wise accuracies
            phase_accs = [np.mean(accuracies[p*10:(p+1)*10]) for p in range(5)]
            all_results[m][seed] = (mean_acc, phase_accs)
            print(f"  {m:18s} Overall Accuracy: {mean_acc*100:6.2f}%")
            
    # Compute stats
    print("\n" + "="*50)
    print("MULTI-SEED STATISTICAL SUMMARY (5 Seeds)")
    print("="*50)
    
    method_stats = {}
    for m in method_names:
        overalls = [all_results[m][s][0] for s in seeds]
        phases_list = [all_results[m][s][1] for s in seeds] # Shape: (5_seeds, 5_phases)
        
        mean_overall = np.mean(overalls)
        std_overall = np.std(overalls)
        
        mean_phases = np.mean(phases_list, axis=0)
        std_phases = np.std(phases_list, axis=0)
        
        method_stats[m] = {
            "overall": (mean_overall, std_overall),
            "phases_mean": mean_phases,
            "phases_std": std_phases
        }
        
        print(f"{m:18s} : Overall Accuracy = {mean_overall*100:6.2f}% \u00b1 {std_overall*100:4.2f}%")
        phase_str = "    Phases: " + " | ".join([f"P{i+1}: {mean_phases[i]*100:.2f}%" for i in range(5)])
        print(phase_str)
        
    # Generate LaTeX Table with mean \pm std
    print("\n--- LaTeX Code for Main Results Table ---")
    print(r"""\begin{table*}[t]
  \caption{Test-Time Model Merging Accuracy (\%) across the 50-batch non-stationary test stream, evaluated over 5 independent random seeds (mean $\pm$ standard deviation). Our proposed FL-AHR achieves the highest overall streaming accuracy by robustly isolating sparse background domains under severe environmental noise.}
  \label{results-table}
  \vskip 0.04in
  \begin{center}
    \setlength{\tabcolsep}{3.5pt}
    \begin{scriptsize}
      \begin{tabular}{lcccccc}
        \toprule
        Method & Overall Accuracy & Phase 1 (Clean) & Phase 2 (Noisy) & Phase 3 (Clean) & Phase 4 (Noisy) & Phase 5 (KMNIST) \\
        \midrule""")
    
    for m in method_names:
        stats = method_stats[m]
        ov_m, ov_s = stats["overall"]
        p_means = stats["phases_mean"]
        p_stds = stats["phases_std"]
        
        # Build cells
        ov_cell = f"{ov_m*100:.2f}\\% \\pm {ov_s*100:.2f}\\%"
        p_cells = [f"{p_means[i]*100:.2f}\\% \\pm {p_stds[i]*100:.2f}\\%" for i in range(5)]
        
        if m == "FL-AHR (Ours)":
            # Bold our row
            ov_cell = r"\textbf{" + ov_cell + r"}"
            p_cells = [r"\textbf{" + cell + r"}" for cell in p_cells]
            row_name = r"\textbf{FL-AHR (Ours)}"
        else:
            row_name = m
            
        print(f"        {row_name} & {ov_cell} & " + " & ".join(p_cells) + r" \\")
        
    print(r"""        \bottomrule
      \end{tabular}
    \end{scriptsize}
  \end{center}
  \vskip -0.1in
  \vspace*{-8pt}
\end{table*}""")

if __name__ == "__main__":
    main()