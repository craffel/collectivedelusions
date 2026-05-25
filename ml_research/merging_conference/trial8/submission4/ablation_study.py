import torch
import numpy as np
import os
import sys
import torch.nn.functional as F

# Ensure path is set up
sys.path.append(os.path.abspath("src"))

from data import download_datasets, get_expert_loaders, build_streams
from merging import run_frtr_ttmm

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Ablation Studies on: {device}")
    
    # Load expert state dicts
    mnist_expert = torch.load("checkpoints/mnist_expert.pt", map_location=device)
    kmnist_expert = torch.load("checkpoints/kmnist_expert.pt", map_location=device)
    experts = [mnist_expert, kmnist_expert]
    
    # Load datasets and build streams
    datasets_dict = download_datasets()
    expert_loaders = get_expert_loaders(datasets_dict)
    
    # Default parameters
    default_kwargs = {
        "device": device,
        "lr": 1e-2,
        "inner_steps": 3,
        "gamma": 15.0,
        "beta": 1.5,
        "alpha_ema": 0.9
    }
    
    # ------------------ Ablation 1: Varying beta (Teacher KL Regularization) ------------------
    print("\n--- Ablation 1: Varying beta (KL weight) ---")
    betas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    ablation1_results = {}
    for b in betas:
        set_seed(42)
        streams = build_streams(expert_loaders)
        kwargs = default_kwargs.copy()
        kwargs["beta"] = b
        
        # We'll evaluate on Closed Sequential and Noisy Open-World
        seq_acc, _ = run_frtr_ttmm(experts, streams["Closed Sequential"], **kwargs)
        noisy_acc, _ = run_frtr_ttmm(experts, streams["Noisy Open-World"], **kwargs)
        ablation1_results[b] = (seq_acc, noisy_acc)
        print(f"beta = {b:3.1f} | Closed Seq: {seq_acc:.4f} | Noisy OW: {noisy_acc:.4f}")
        
    # ------------------ Ablation 2: Varying alpha_ema (EMA Momentum) ------------------
    print("\n--- Ablation 2: Varying alpha_ema (EMA Momentum) ---")
    alphas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
    ablation2_results = {}
    for a in alphas:
        set_seed(42)
        streams = build_streams(expert_loaders)
        kwargs = default_kwargs.copy()
        kwargs["alpha_ema"] = a
        
        seq_acc, _ = run_frtr_ttmm(experts, streams["Closed Sequential"], **kwargs)
        noisy_acc, _ = run_frtr_ttmm(experts, streams["Noisy Open-World"], **kwargs)
        ablation2_results[a] = (seq_acc, noisy_acc)
        print(f"alpha_ema = {a:4.2f} | Closed Seq: {seq_acc:.4f} | Noisy OW: {noisy_acc:.4f}")

    # ------------------ Ablation 3: Varying gamma (MI Scaling) ------------------
    print("\n--- Ablation 3: Varying gamma (MI Scaling) ---")
    gammas = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    ablation3_results = {}
    for g in gammas:
        set_seed(42)
        streams = build_streams(expert_loaders)
        kwargs = default_kwargs.copy()
        kwargs["gamma"] = g
        
        seq_acc, _ = run_frtr_ttmm(experts, streams["Closed Sequential"], **kwargs)
        noisy_acc, _ = run_frtr_ttmm(experts, streams["Noisy Open-World"], **kwargs)
        ablation3_results[g] = (seq_acc, noisy_acc)
        print(f"gamma = {g:4.1f} | Closed Seq: {seq_acc:.4f} | Noisy OW: {noisy_acc:.4f}")

    # Print LaTeX tables
    print("\n=== LATEX TABLES FOR PAPER ===")
    
    print("\n% Ablation 1: beta")
    print("\\begin{table}[h]")
    print("\\caption{Ablation study on the strength of teacher regularization ($\\beta$).}")
    print("\\label{tab:ablation_beta}")
    print("\\centering")
    print("\\begin{tabular}{ccc}")
    print("\\toprule")
    print("$\\beta$ & Closed Sequential & Noisy Open-World \\\\")
    print("\\midrule")
    for b, (seq, noisy) in ablation1_results.items():
        bold_seq = f"\\textbf{{{seq:.4f}}}" if b == 1.5 else f"{seq:.4f}"
        bold_noisy = f"\\textbf{{{noisy:.4f}}}" if b == 1.5 else f"{noisy:.4f}"
        print(f"{b:3.1f} & {bold_seq} & {bold_noisy} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n% Ablation 2: alpha_ema")
    print("\\begin{table}[h]")
    print("\\caption{Ablation study on the EMA teacher momentum ($\\alpha_{\\text{ema}}$).}")
    print("\\label{tab:ablation_alpha}")
    print("\\centering")
    print("\\begin{tabular}{ccc}")
    print("\\toprule")
    print("$\\alpha_{\\text{ema}}$ & Closed Sequential & Noisy Open-World \\\\")
    print("\\midrule")
    for a, (seq, noisy) in ablation2_results.items():
        bold_seq = f"\\textbf{{{seq:.4f}}}" if a == 0.9 else f"{seq:.4f}"
        bold_noisy = f"\\textbf{{{noisy:.4f}}}" if a == 0.9 else f"{noisy:.4f}"
        print(f"{a:4.2f} & {bold_seq} & {bold_noisy} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n% Ablation 3: gamma")
    print("\\begin{table}[h]")
    print("\\caption{Ablation study on the Mutual Information scaling parameter ($\\gamma$).}")
    print("\\label{tab:ablation_gamma}")
    print("\\centering")
    print("\\begin{tabular}{ccc}")
    print("\\toprule")
    print("$\\gamma$ & Closed Sequential & Noisy Open-World \\\\")
    print("\\midrule")
    for g, (seq, noisy) in ablation3_results.items():
        bold_seq = f"\\textbf{{{seq:.4f}}}" if g == 15.0 else f"{seq:.4f}"
        bold_noisy = f"\\textbf{{{noisy:.4f}}}" if g == 15.0 else f"{noisy:.4f}"
        print(f"{g:4.1f} & {bold_seq} & {bold_noisy} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
