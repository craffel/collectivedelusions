import matplotlib.pyplot as plt
import numpy as np

# Set style parameters for academic plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Define professional color palette (Seaborn-like)
COLOR_UNICAL = '#D95F02'  # Orange-red
COLOR_HNS = '#1F78B4'    # Classic deep blue
COLOR_ORACLE = '#33A02C' # Muted green

def plot_comparison():
    fig, ax = plt.subplots(figsize=(6, 3.8))
    
    methods = ['Weight\nAveraging', 'Task Arithmetic\n(\\lambda=0.5)', 'TIES-Merging', 'DARE-Merging']
    uncalibrated = [29.91, 42.01, 18.50, 15.58]
    hns_calibrated = [43.73, 43.73, 36.04, 34.12]
    
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, uncalibrated, width, label='Uncalibrated Baseline', color=COLOR_UNICAL, edgecolor='black', linewidth=0.5, hatch='//')
    rects2 = ax.bar(x + width/2, hns_calibrated, width, label='HNS (Ours)', color=COLOR_HNS, edgecolor='black', linewidth=0.5)
    
    # Add horizontal line for average expert oracle performance
    ax.axhline(y=90.91, color=COLOR_ORACLE, linestyle='--', linewidth=1.5, label='Expert Oracle (90.91%)')
    
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Multi-Task Model Merging Performance Recovery')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('plot_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Generated plot_comparison.pdf")

def plot_partial_merging():
    fig, ax = plt.subplots(figsize=(6, 3.8))
    
    configs = ['Merge\nAll', 'Merge Only\nLayer 1 & 2', 'Merge Only\nLayer 3 & 4', 'Merge All\nexcept Layer 4']
    uncalibrated = [29.91, 68.57, 56.35, 63.01]
    hns_calibrated = [43.73, 75.88, 67.93, 71.75]
    
    x = np.arange(len(configs))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, uncalibrated, width, label='WA Baseline', color=COLOR_UNICAL, edgecolor='black', linewidth=0.5, hatch='\\\\')
    rects2 = ax.bar(x + width/2, hns_calibrated, width, label='HNS (Ours)', color=COLOR_HNS, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=90.91, color=COLOR_ORACLE, linestyle='--', linewidth=1.5, label='Expert Oracle (90.91%)')
    
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('HNS vs. Uncalibrated across Partial Merging Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('plot_partial_merging.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Generated plot_partial_merging.pdf")

def plot_similarities():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    
    datasets = ['MNIST', 'Fashion-MNIST', 'CIFAR-10']
    raw_weights = [0.9934, 0.9921, 0.9913]
    task_vectors = [0.5134, 0.5994, 0.6358]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, raw_weights, width, label='Raw Weight Tensors', color='#7F7F7F', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, task_vectors, width, label='Task-Vector Updates', color=COLOR_HNS, edgecolor='black', linewidth=0.5, hatch='..')
    
    # Theoretical orthogonality line for K=3
    ax.axhline(y=0.5774, color='red', linestyle=':', linewidth=1.5, label='Theoretical Orthogonality (1/\\sqrt{3} \\approx 0.577)')
    
    ax.set_ylabel('Average Cosine Similarity')
    ax.set_title('Layer-Wise Cosine Similarity with Merged Model')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('plot_similarities.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Generated plot_similarities.pdf")

if __name__ == '__main__':
    plot_comparison()
    plot_partial_merging()
    plot_similarities()
