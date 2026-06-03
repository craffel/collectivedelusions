import matplotlib.pyplot as plt
import numpy as np

# Set style for professional publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif']
plt.rcParams['text.usetex'] = False  # Avoid requiring full latex install in this script environment
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

def plot_pruning_comparison():
    # X-axis (Pruning Ratio in %)
    ratios = np.array([10, 30, 50])
    
    # Data under INT4
    random = [64.62, 43.67, 30.29]
    weight_l1 = [70.45, 57.64, 29.36]
    joint_acp = [70.29, 49.73, 31.60]
    dynamic_acp_l1 = [71.70, 58.70, 41.85]
    dynamic_acp_var = [71.71, 69.78, 52.27]
    
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    
    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Plot lines
    ax.plot(ratios, random, marker='o', linestyle=':', color='#7f7f7f', linewidth=1.5, markersize=6, label='Random Pruning', zorder=1)
    ax.plot(ratios, weight_l1, marker='s', linestyle='--', color='#d62728', linewidth=1.5, markersize=6, label='Weight L1 (Baseline)', zorder=2)
    ax.plot(ratios, joint_acp, marker='^', linestyle='-.', color='#17becf', linewidth=1.5, markersize=6, label='Joint ACP (L1)', zorder=3)
    ax.plot(ratios, dynamic_acp_l1, marker='D', linestyle='-', color='#bcbd22', linewidth=1.5, markersize=6, label='Dynamic ACP (L1)', zorder=4)
    ax.plot(ratios, dynamic_acp_var, marker='*', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=9, label='Dynamic ACP (Variance - Ours)', zorder=5)
    
    ax.set_xlabel('Pruning Ratio (%)')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Pruning Performance under INT4 Quantization')
    ax.set_xticks(ratios)
    ax.set_ylim(20, 80)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False)
    
    plt.tight_layout()
    plt.savefig('pruning_accuracy_int4.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Pruning comparison plot saved successfully as pruning_accuracy_int4.pdf")

def plot_calibration_sweep():
    # X-axis (Calibration Set Size N)
    n_sizes = np.array([16, 32, 64, 128])
    
    # Data under INT4 at 30% pruning
    dynamic_acp_l1 = [47.80, 53.25, 58.70, 61.31]
    dynamic_acp_var = [60.23, 66.19, 69.78, 70.01]
    
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    
    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Plot lines
    ax.plot(n_sizes, dynamic_acp_l1, marker='D', linestyle='--', color='#bcbd22', linewidth=2.0, markersize=7, label='Dynamic ACP (L1)', zorder=1)
    ax.plot(n_sizes, dynamic_acp_var, marker='*', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=10, label='Dynamic ACP (Variance - Ours)', zorder=2)
    
    # Highlight N=64 as the optimal point
    ax.axvline(x=64, color='#d62728', linestyle=':', alpha=0.8, linewidth=1.5)
    ax.text(66, 50, 'Optimal N=64', color='#d62728', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Calibration Set Size (N per task)')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Sensitivity to Calibration Set Size N (INT4, 30% Pruning)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(n_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(40, 75)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False)
    
    plt.tight_layout()
    plt.savefig('calibration_set_size_sweep.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Calibration sweep plot saved successfully as calibration_set_size_sweep.pdf")

if __name__ == '__main__':
    plot_pruning_comparison()
    plot_calibration_sweep()
