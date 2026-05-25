import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("src"))

from data import download_datasets, get_expert_loaders, build_streams
from merging import run_static_merging, run_adamerging, run_df_bayes_ttmm, run_frtr_ttmm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running multi-seed evaluation on: {device}")
    
    # Load experts
    mnist_expert = torch.load("checkpoints/mnist_expert.pt", map_location=device)
    kmnist_expert = torch.load("checkpoints/kmnist_expert.pt", map_location=device)
    experts = [mnist_expert, kmnist_expert]
    
    # Load datasets
    datasets_dict = download_datasets()
    expert_loaders = get_expert_loaders(datasets_dict)
    
    seeds = [42, 43, 44, 45, 46]
    results = {
        "Static Merging": [],
        "AdaMerging": [],
        "DF-Bayes-TTMM": [],
        "MI-FRTR-TTMM (Ours)": []
    }
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        set_seed(seed)
        streams = build_streams(expert_loaders)
        noisy_stream = streams["Noisy Open-World"]
        
        # 1. Static Merging
        set_seed(seed)
        static_acc, _ = run_static_merging(experts, noisy_stream, device=device)
        results["Static Merging"].append(static_acc)
        
        # 2. AdaMerging
        set_seed(seed)
        ada_acc, _ = run_adamerging(experts, noisy_stream, device=device)
        results["AdaMerging"].append(ada_acc)
        
        # 3. DF-Bayes-TTMM
        set_seed(seed)
        bayes_acc, _ = run_df_bayes_ttmm(experts, noisy_stream, device=device)
        results["DF-Bayes-TTMM"].append(bayes_acc)
        
        # 4. MI-FRTR-TTMM (Ours)
        set_seed(seed)
        our_acc, _ = run_frtr_ttmm(experts, noisy_stream, device=device)
        results["MI-FRTR-TTMM (Ours)"].append(our_acc)
        
        print(f"Static Merging: {static_acc:.4f} | AdaMerging: {ada_acc:.4f} | DF-Bayes-TTMM: {bayes_acc:.4f} | MI-FRTR-TTMM: {our_acc:.4f}")
        
    print("\n=== Final Multi-Seed Results (Noisy Open-World) ===")
    for method, accs in results.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{method:20s} | Mean: {mean_acc:.4f} | Std: {std_acc:.4f} | Values: {[round(x, 4) for x in accs]}")

if __name__ == "__main__":
    main()
