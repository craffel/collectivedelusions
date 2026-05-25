import matplotlib.pyplot as plt
import numpy as np

# Set professional font and style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'text.usetex': False
})

def plot_results_comparison():
    # Data from Table 1
    categories = ['Clean', 'Noise', 'Blur', 'Contrast', 'Rotation']
    methods = {
        'SyMerge': [72.03, 14.82, 70.71, 41.12, 40.62],
        'SAT-SyMerge': [70.05, 15.41, 69.37, 40.55, 42.10],
        'ASAM-SyMerge': [70.73, 15.35, 68.88, 41.52, 40.23],
        'SBF-Global (Ours)': [68.99, 16.46, 69.72, 39.83, 34.26],
        'SBF-Per-Tensor (Ours)': [72.74, 11.72, 71.50, 42.33, 37.29]
    }
    
    colors = ['#cccccc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
    
    x = np.arange(len(categories))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    
    for i, (method_name, scores) in enumerate(methods.items()):
        rects = ax.bar(x + (i - 2) * width, scores, width, label=method_name, color=colors[i], edgecolor='black', linewidth=0.5)
        
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Performance Comparison Across Environments', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 90)
    ax.legend(frameon=True, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results_comparison.pdf', format='pdf', dpi=300)
    plt.close()
    print("Generated results_comparison.pdf")

def plot_beta_tradeoff():
    # Data from Table 2
    betas = [0.0, 0.5, 0.9, 0.99]
    clean_acc = [72.47, 72.47, 72.74, 72.47]
    noise_acc = [13.69, 14.09, 11.72, 12.73]
    ood_avg = [41.98, 42.08, 40.71, 41.75]
    
    fig, ax1 = plt.subplots(figsize=(6, 2.7))
    
    # Left axis for Clean and OOD Average
    color = '#1f77b4'
    ax1.set_xlabel('Fisher Momentum Parameter ($\\beta$)', fontweight='bold')
    ax1.set_ylabel('Clean & OOD Avg Accuracy (%)', color=color, fontweight='bold')
    line1 = ax1.plot(betas, clean_acc, 'o-', color='#1f77b4', linewidth=2, label='Clean Accuracy')
    line2 = ax1.plot(betas, ood_avg, 's--', color='#41b6c4', linewidth=2, label='OOD Average')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(35, 80)
    
    # Right axis for Noise
    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Gaussian Noise Accuracy (%)', color=color, fontweight='bold')
    line3 = ax2.plot(betas, noise_acc, '^-.', color='#d62728', linewidth=2, label='Noise Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(10, 16)
    
    # Put legends together
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', frameon=True)
    
    plt.title('Impact of Momentum $\\beta$ on Robustness vs. Clean Accuracy', fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('beta_tradeoff.pdf', format='pdf', dpi=300)
    plt.close()
    print("Generated beta_tradeoff.pdf")

if __name__ == '__main__':
    plot_results_comparison()
    plot_beta_tradeoff()
