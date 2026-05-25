import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    # Style settings
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.titlesize': 12
    })
    
    # Data for FW-SPOR (beta = 0.20)
    gammas = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    acc_ta = np.array([83.33, 83.08, 82.88, 82.77, 82.75])
    acc_co = np.array([76.29, 81.98, 83.14, 79.09, 81.25])
    acc_om = np.array([71.12, 76.64, 74.49, 73.66, 73.26])
    
    # Baselines
    sam_ta, sam_co, sam_om = 82.15, 82.98, 74.31
    sgd_ta, sgd_co, sgd_om = 84.08, 82.85, 78.18
    
    # Plot 1: Full Accuracy vs. Gamma
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    # Draw curves
    ax.plot(gammas, acc_co, 'o-', color='#d62728', linewidth=2.0, label='FW-SPOR + C-Ortho (Ours)', markersize=6)
    ax.plot(gammas, acc_ta, 's-', color='#1f77b4', linewidth=1.5, label='FW-SPOR + Task Arithmetic', markersize=5)
    ax.plot(gammas, acc_om, 'd-', color='#2ca02c', linewidth=1.5, label='FW-SPOR + OM-All', markersize=5)
    
    # Draw baseline dashed lines (C-Ortho only for simplicity to avoid clutter, or draw SAM C-Ortho)
    ax.axhline(y=sam_co, color='#d62728', linestyle='--', alpha=0.6, label='SAM + C-Ortho Baseline')
    ax.axhline(y=sgd_co, color='gray', linestyle=':', alpha=0.6, label='SGD + C-Ortho Baseline')
    
    # Labels and grid
    ax.set_xlabel(r'Temperature Hyperparameter $\gamma$ (Standard SPOR at $\gamma=0.0$)', fontsize=10)
    ax.set_ylabel('Full CIFAR-10 Accuracy (%)', fontsize=10)
    ax.set_title('Merging Performance vs. Temperature Sensitivity Weighting', pad=10)
    ax.set_xticks(gammas)
    ax.set_xticklabels([r'0.0 (SPOR)', '0.1', '0.5 (Best)', '1.0', '2.0'])
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Annotate optimal performance
    ax.annotate(f'Best: 83.14%', xy=(0.5, 83.14), xytext=(0.8, 83.5),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                fontsize=8, fontweight='bold')
    
    # Place legend
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')
    
    # Save plot as PDF/PNG
    plt.tight_layout()
    plt.savefig('template/gamma_accuracy.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('template/gamma_accuracy.png', format='png', bbox_inches='tight', dpi=300)
    print("Successfully generated template/gamma_accuracy.pdf and template/gamma_accuracy.png")

if __name__ == "__main__":
    generate_plots()
