import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    if not os.path.exists("sweep_results.json"):
        print("sweep_results.json not found!")
        return
        
    with open("sweep_results.json", "r") as f:
        data = json.load(f)
        
    rho_sweep = data["rho_sweep"]
    lr_sweep = data["lr_sweep"]
    
    shots = [4, 16, 64, 256]
    
    # 1. Plot rho sweep
    rho_values = sorted([float(k) for k in rho_sweep.keys()])
    plt.figure(figsize=(8, 5))
    for shot in shots:
        means = []
        errs = []
        for rho in rho_values:
            vals = rho_sweep[str(rho)][str(shot)]
            means.append(np.mean(vals))
            errs.append(np.std(vals) / np.sqrt(len(vals)))
        plt.errorbar(rho_values, means, yerr=errs, label=f"{shot} shots", marker="o", capsize=3)
        
    plt.xlabel("Perturbation scale (rho)", fontsize=12)
    plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=12)
    plt.title("Effect of Perturbation Scale (rho) on SA-TTA KL", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("rho_sensitivity.png", dpi=300)
    print("Saved rho_sensitivity.png")
    
    # 2. Plot lr sweep
    lr_values = sorted([float(k) for k in lr_sweep.keys()])
    plt.figure(figsize=(8, 5))
    for shot in shots:
        means = []
        errs = []
        for lr in lr_values:
            vals = lr_sweep[str(lr)][str(shot)]
            means.append(np.mean(vals))
            errs.append(np.std(vals) / np.sqrt(len(vals)))
        plt.errorbar(lr_values, means, yerr=errs, label=f"{shot} shots", marker="o", capsize=3)
        
    plt.xscale("log")
    plt.xlabel("Learning rate", fontsize=12)
    plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=12)
    plt.title("Effect of Test-Time Learning Rate on SA-TTA KL", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("lr_sensitivity.png", dpi=300)
    print("Saved lr_sensitivity.png")

    # Generate LaTeX code for sensitivity tables
    print("\n--- LaTeX Table for Rho Sensitivity (SA-TTA KL) ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Sensitivity of SA-TTA KL to the perturbation scale $\\rho$.}")
    print("\\label{tab:rho_sensitivity}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("$\\rho$ & 4 shots & 16 shots & 64 shots & 256 shots \\\\")
    print("\\midrule")
    for rho in rho_values:
        row = f"{rho}"
        for shot in shots:
            vals = rho_sweep[str(rho)][str(shot)]
            row += f" & {np.mean(vals):.2f} \\pm {np.std(vals)/np.sqrt(len(vals)):.2f}"
        row += " \\\\"
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n--- LaTeX Table for LR Sensitivity (SA-TTA KL) ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Sensitivity of SA-TTA KL to the learning rate $\\eta$ (at $\\rho=0.1$).}")
    print("\\label{tab:lr_sensitivity}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("$\\eta$ & 4 shots & 16 shots & 64 shots & 256 shots \\\\")
    print("\\midrule")
    for lr in lr_values:
        row = f"{lr}"
        for shot in shots:
            vals = lr_sweep[str(lr)][str(shot)]
            row += f" & {np.mean(vals):.2f} \\pm {np.std(vals)/np.sqrt(len(vals)):.2f}"
        row += " \\\\"
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
