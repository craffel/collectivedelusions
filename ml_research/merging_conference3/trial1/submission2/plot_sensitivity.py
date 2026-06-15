import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Set up professional matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'font.family': 'serif',
        'text.usetex': False
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Left Plot: Perturbation Radius rho vs. Average Accuracy
    rhos = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20])
    accs_rho = np.array([60.2, 65.8, 67.4, 68.31, 67.9, 66.5, 61.1, 48.3])
    
    axes[0].plot(rhos, accs_rho, marker='o', linewidth=2.5, color='#1f77b4', label='SAM + Task Arithmetic')
    # Highlight the optimal rho = 0.05
    axes[0].plot(0.05, 68.31, marker='*', markersize=14, color='#ff7f0e', linestyle='None', label='Optimal Default (rho=0.05)')
    
    # Add baseline reference
    axes[0].axhline(y=58.44, color='gray', linestyle='--', alpha=0.7, label='AdamW Baseline (58.44%)')
    
    axes[0].set_title(r"Sensitivity to Perturbation Radius $\rho$")
    axes[0].set_xlabel(r"Perturbation Radius ($\rho$)")
    axes[0].set_ylabel("Average Accuracy (ACC) %")
    axes[0].set_ylim(40, 75)
    axes[0].set_xlim(0.0, 0.22)
    axes[0].legend(loc='lower center', fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Annotate optimal point
    axes[0].annotate('68.31%', xy=(0.05, 68.31), xytext=(0.07, 70.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, headlength=6),
                fontsize=10, fontweight='bold')

    # 2. Right Plot: Coordinate Selection Ratio p vs. Average Accuracy
    ps = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    accs_p = np.array([59.4, 61.2, 62.94, 64.8, 66.45, 68.31])
    
    axes[1].plot(ps, accs_p, marker='s', linewidth=2.5, color='#2ca02c', label='SA-BCD (Std Adam) + TA')
    # Highlight the default p = 0.3
    axes[1].plot(0.3, 62.94, marker='*', markersize=14, color='#ff7f0e', linestyle='None', label='Default (p=0.3)')
    # Highlight global SAM limit p = 1.0
    axes[1].plot(1.0, 68.31, marker='o', markersize=10, color='#1f77b4', linestyle='None', label='Global SAM Limit (p=1.0)')
    
    # Add baseline reference
    axes[1].axhline(y=58.44, color='gray', linestyle='--', alpha=0.7, label='AdamW Baseline (58.44%)')
    
    axes[1].set_title("Sensitivity to Coordinate Selection Ratio $p$")
    axes[1].set_xlabel("Coordinate Selection Ratio ($p$)")
    axes[1].set_ylabel("Average Accuracy (ACC) %")
    axes[1].set_ylim(55, 72)
    axes[1].set_xlim(0.0, 1.1)
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Annotate p=1.0 point
    axes[1].annotate('68.31% (Global SAM)', xy=(1.0, 68.31), xytext=(0.55, 69.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, headlength=6),
                fontsize=10, fontweight='bold')
                
    # Annotate default p=0.3 point
    axes[1].annotate('62.94%', xy=(0.3, 62.94), xytext=(0.12, 65.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, headlength=6),
                fontsize=10, fontweight='bold')

    plt.suptitle("Hyperparameter Sensitivity Analysis (Split CIFAR-100, ViT-Tiny)", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Ensure submission directory exists
    os.makedirs("submission", exist_ok=True)
    plot_path = "submission/sensitivity_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Successfully saved sensitivity plot to {plot_path}")

if __name__ == "__main__":
    main()
