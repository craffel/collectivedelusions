import matplotlib.pyplot as plt
import numpy as np

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    'figure.titlesize': 14
})

# Color palette
colors = {
    'spja_tta': '#2ca02c', # green
    'spja_sft': '#1f77b4', # blue
    'ntaac': '#ff7f0e',    # orange
    'none': '#d62728',     # red
    'lsc': '#9467bd'       # purple
}

def plot_convergence():
    epochs = [5, 10, 15, 20]
    # SFT average accuracies
    sft_accs = [74.7067, 75.6367, 76.1300, 75.7300]
    # TTA average accuracies
    tta_accs = [74.8033, 76.0867, 76.7833, 76.6567]
    
    plt.figure(figsize=(5, 3.5))
    plt.plot(epochs, tta_accs, marker='o', linestyle='-', color=colors['spja_tta'], linewidth=2, label='SPJA-TTA')
    plt.plot(epochs, sft_accs, marker='s', linestyle='--', color=colors['spja_sft'], linewidth=2, label='SPJA-SFT')
    
    plt.xlabel('Head-adaptation Epochs ($E$)')
    plt.ylabel('Average Accuracy (%)')
    plt.xticks(epochs)
    plt.ylim(74.0, 77.5)
    plt.title('Convergence Analysis')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig('fig_convergence.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig_convergence.pdf")

def plot_lambda_sweep():
    lambdas = [0.1, 0.2, 0.3, 0.4]
    ntaac_accs = [44.6933, 64.3167, 71.4367, 73.9367]
    spja_tta_accs = [58.7700, 70.1467, 75.8000, 77.6233]
    none_accs = [28.8033, 40.3600, 37.9533, 9.9333]
    
    plt.figure(figsize=(5, 3.5))
    plt.plot(lambdas, spja_tta_accs, marker='o', linestyle='-', color=colors['spja_tta'], linewidth=2, label='SPJA-TTA (Ours)')
    plt.plot(lambdas, ntaac_accs, marker='^', linestyle='--', color=colors['ntaac'], linewidth=2, label='N-TAAC')
    plt.plot(lambdas, none_accs, marker='x', linestyle=':', color=colors['none'], linewidth=1.5, label='NONE (Uncalibrated)')
    
    plt.xlabel('Task Arithmetic Scaling ($\lambda$)')
    plt.ylabel('Average Accuracy (%)')
    plt.xticks(lambdas)
    plt.ylim(5.0, 85.0)
    plt.title('Task Arithmetic $\lambda$ Sweep')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig('fig_lambda_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig_lambda_sweep.pdf")

def plot_sample_complexity():
    sizes = [32, 128, 512]
    # Data from calibration strategies
    # Balanced
    balanced_tta = [74.7000, 77.2600, 78.5700]
    balanced_sft = [72.3300, 76.5100, 78.3100]
    # Sequential
    seq_tta = [72.5167, 76.7067, 78.5400]
    # Random
    rand_tta = [72.9500, 76.8400, 78.5000]
    
    plt.figure(figsize=(5, 3.5))
    plt.plot(sizes, balanced_tta, marker='o', linestyle='-', color=colors['spja_tta'], linewidth=2, label='Class-Balanced')
    plt.plot(sizes, rand_tta, marker='s', linestyle='--', color=colors['spja_sft'], linewidth=2, label='Random')
    plt.plot(sizes, seq_tta, marker='^', linestyle='-.', color=colors['ntaac'], linewidth=2, label='Sequential (First-N)')
    
    plt.xlabel('Calibration Samples per Task ($N$)')
    plt.ylabel('Average Accuracy (%)')
    plt.xscale('log')
    plt.xticks(sizes, labels=['32', '128', '512'])
    plt.ylim(71.0, 79.5)
    plt.title('Calibration Selection Strategy')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig('fig_cal_strategy.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig_cal_strategy.pdf")

if __name__ == '__main__':
    plot_convergence()
    plot_lambda_sweep()
    plot_sample_complexity()
