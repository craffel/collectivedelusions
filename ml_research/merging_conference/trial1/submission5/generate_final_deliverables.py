import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Main results dictionary
r = {
    "Expert_Clean": {"cifar10": 54.00, "svhn": 44.00, "mnist": 94.80},
    "Expert_Corrupt": {"cifar10": 17.20, "svhn": 13.40, "mnist": 13.80},
    "WA_Clean": {"cifar10": 25.80, "svhn": 23.40, "mnist": 29.60},
    "WA_Corrupt": {"cifar10": 10.20, "svhn": 10.40, "mnist": 18.00},
    "TA_Clean": {"cifar10": 28.60, "svhn": 23.20, "mnist": 25.60},
    "TA_Corrupt": {"cifar10": 13.00, "svhn": 6.60, "mnist": 19.60},
    "OM_Clean": {"cifar10": 25.40, "svhn": 23.20, "mnist": 11.20},
    "OM_Corrupt": {"cifar10": 16.20, "svhn": 6.40, "mnist": 9.80},
    "SyMerge_Clean": {"cifar10": 53.00, "svhn": 41.60, "mnist": 88.60},
    "SyMerge_Corrupt": {"cifar10": 13.40, "svhn": 14.80, "mnist": 11.20},
    "SynOrtho_Clean": {"cifar10": 56.80, "svhn": 44.60, "mnist": 83.80},
    "SynOrtho_Corrupt": {"cifar10": 16.40, "svhn": 17.60, "mnist": 13.00},
    "SynOrtho_NoAdapt_Clean": {"cifar10": 25.40, "svhn": 23.20, "mnist": 11.20},
    "SynOrtho_NoAdapt_Corrupt": {"cifar10": 15.00, "svhn": 8.20, "mnist": 9.80},
    "SynOrtho_NoCoeff_Clean": {"cifar10": 56.80, "svhn": 44.60, "mnist": 83.80},
    "SynOrtho_NoCoeff_Corrupt": {"cifar10": 16.60, "svhn": 17.00, "mnist": 11.60}
}

# Sensitivity results list: lr, steps, c10, svhn, mnist, avg
sensitivity_data = [
    (1e-4, 10, 16.33, 15.13, 15.13, 15.53),
    (1e-4, 50, 17.13, 15.93, 15.93, 16.33),
    (1e-4, 100, 15.93, 14.73, 14.73, 15.13),
    (1e-3, 10, 13.87, 12.67, 12.67, 13.07),
    (1e-3, 50, 16.60, 15.40, 15.40, 15.80),
    (1e-3, 100, 15.47, 14.27, 14.27, 14.67),
    (5e-3, 10, 16.33, 15.13, 15.13, 15.53),
    (5e-3, 50, 16.67, 15.47, 15.47, 15.87),
    (5e-3, 100, 17.00, 16.40, 12.20, 15.20)
]

def save_results_text():
    summary_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("MODEL MERGING PERFORMANCE SUMMARY\n")
        f.write("===============================\n\n")
        
        for cond in ["Clean", "Corrupt"]:
            f.write(f"--- {cond.upper()} EVALUATION ---\n")
            f.write(f"{'Method':<35} | {'CIFAR-10':<10} | {'SVHN':<10} | {'MNIST':<10} | {'Average':<10}\n")
            f.write("-" * 80 + "\n")
            
            methods = [
                ("Single Expert", f"Expert_{cond}"),
                ("Weight Averaging", f"WA_{cond}"),
                ("Task Arithmetic", f"TA_{cond}"),
                ("OrthoMerge", f"OM_{cond}"),
                ("SyMerge (TT)", f"SyMerge_{cond}"),
                ("SynOrtho (Ours)", f"SynOrtho_{cond}"),
                ("SynOrtho Ablation: No Adapter", f"SynOrtho_NoAdapt_{cond}"),
                ("SynOrtho Ablation: No Coeff", f"SynOrtho_NoCoeff_{cond}")
            ]
            
            for label, key in methods:
                c10 = r[key]["cifar10"]
                svhn = r[key]["svhn"]
                mnist = r[key]["mnist"]
                avg = (c10 + svhn + mnist) / 3
                f.write(f"{label:<35} | {c10:<10.2f} | {svhn:<10.2f} | {mnist:<10.2f} | {avg:<10.2f}\n")
            f.write("\n")
            
    print(f"Saved text summary to {summary_path}")

def save_sensitivity_text():
    sens_path = os.path.join(RESULTS_DIR, "sensitivity_analysis.txt")
    with open(sens_path, "w") as f:
        f.write("HYPERPARAMETER SENSITIVITY SWEEP (CORRUPTED OOD EVALUATION)\n")
        f.write("=========================================================\n\n")
        f.write(f"{'Learning Rate':<15} | {'Adapt Steps':<12} | {'CIFAR-10':<10} | {'SVHN':<10} | {'MNIST':<10} | {'Average':<10}\n")
        f.write("-" * 80 + "\n")
        for lr, steps, c10, svhn, mnist, avg in sensitivity_data:
            f.write(f"{lr:<15.1e} | {steps:<12d} | {c10:<10.2f} | {svhn:<10.2f} | {mnist:<10.2f} | {avg:<10.2f}\n")
    print(f"Saved sensitivity sweep results to {sens_path}")

def save_plots():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    tasks_labels = ["CIFAR-10", "SVHN", "MNIST", "Average"]
    methods_keys = [
        ("Weight Averaging", "WA"),
        ("Task Arithmetic", "TA"),
        ("OrthoMerge", "OM"),
        ("SyMerge (TT)", "SyMerge"),
        ("SynOrtho (Ours)", "SynOrtho"),
        ("SynOrtho: No Adapter", "SynOrtho_NoAdapt"),
        ("SynOrtho: No Coeff", "SynOrtho_NoCoeff")
    ]
    
    # Clean Plot
    ax = axes[0]
    x = np.arange(len(tasks_labels))
    width = 0.11
    
    for i, (label, key) in enumerate(methods_keys):
        c10 = r[f"{key}_Clean"]["cifar10"]
        svhn = r[f"{key}_Clean"]["svhn"]
        mnist = r[f"{key}_Clean"]["mnist"]
        avg = (c10 + svhn + mnist) / 3
        vals = [c10, svhn, mnist, avg]
        ax.bar(x + (i - 3) * width, vals, width, label=label)
        
    c10_exp = r["Expert_Clean"]["cifar10"]
    svhn_exp = r["Expert_Clean"]["svhn"]
    mnist_exp = r["Expert_Clean"]["mnist"]
    avg_exp = (c10_exp + svhn_exp + mnist_exp) / 3
    ax.plot([0, 1, 2, 3], [c10_exp, svhn_exp, mnist_exp, avg_exp], "k--", label="Individual Experts", alpha=0.7)
    
    ax.set_title("Clean Evaluation Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize='small', loc='lower left')
    
    # Corrupt Plot
    ax = axes[1]
    for i, (label, key) in enumerate(methods_keys):
        c10 = r[f"{key}_Corrupt"]["cifar10"]
        svhn = r[f"{key}_Corrupt"]["svhn"]
        mnist = r[f"{key}_Corrupt"]["mnist"]
        avg = (c10 + svhn + mnist) / 3
        vals = [c10, svhn, mnist, avg]
        ax.bar(x + (i - 3) * width, vals, width, label=label)
        
    c10_exp_c = r["Expert_Corrupt"]["cifar10"]
    svhn_exp_c = r["Expert_Corrupt"]["svhn"]
    mnist_exp_c = r["Expert_Corrupt"]["mnist"]
    avg_exp_c = (c10_exp_c + svhn_exp_c + mnist_exp_c) / 3
    ax.plot([0, 1, 2, 3], [c10_exp_c, svhn_exp_c, mnist_exp_c, avg_exp_c], "k--", label="Individual Experts", alpha=0.7)
    
    ax.set_title("Corrupted (OOD) Evaluation Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize='small', loc='upper right')
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "results_chart.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved bar chart plot to {plot_path}")

if __name__ == "__main__":
    save_results_text()
    save_sensitivity_text()
    save_plots()
