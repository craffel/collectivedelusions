import matplotlib.pyplot as plt
import numpy as np
import json

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

def plot_robustness():
    # Robustness metrics
    scenarios = ['Clean', 'Gauss. Noise\n(Soft)', 'Gauss. Noise\n(Heavy)', 'Brightness\n(+0.15)', 'Translation\n(+2 px)']
    mspr = [97.49, 97.38, 92.54, 97.42, 97.42]
    ihtr = [91.98, 91.77, 94.94, 33.33, 91.98]
    hlbr = [96.30, 96.31, 96.32, 96.23, 93.34]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Elegant minimalist color palette (dark grey, soft red/coral, deep blue)
    rects1 = ax.bar(x - width, mspr, width, label='MSPR (Hidden)', color='#7f8c8d', edgecolor='#2c3e50', alpha=0.9)
    rects2 = ax.bar(x, ihtr, width, label='IHTR (Ours, Baseline)', color='#e74c3c', edgecolor='#c0392b', alpha=0.9)
    rects3 = ax.bar(x + width, hlbr, width, label='HLBR (Ours, Proposed)', color='#3498db', edgecolor='#2980b9', alpha=0.9)
    
    ax.set_ylabel('Task Routing Accuracy (%)', fontweight='semibold')
    ax.set_title('Task Routing Robustness under Environmental Perturbations', fontweight='semibold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='medium')
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig('robustness_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Generated robustness_comparison.pdf")

def plot_sweeps():
    # Data scaling sweep (calibration size N)
    sizes = [16, 64, 128, 256]
    rout_acc = [98.24, 98.23, 98.74, 99.08]
    model_acc = [38.57, 38.79, 38.77, 38.98]
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    color = '#3498db'
    ax1.set_xlabel('Calibration Set Size N per Task', fontweight='semibold')
    ax1.set_ylabel('Routing Accuracy (%)', color=color, fontweight='semibold')
    line1 = ax1.plot(sizes, rout_acc, marker='o', linewidth=2, color=color, label='Routing Accuracy', markersize=6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(97, 100)
    
    # Instantiating a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = '#2ecc71'
    ax2.set_ylabel('Average Merged Accuracy (%)', color=color, fontweight='semibold')
    line2 = ax2.plot(sizes, model_acc, marker='s', linewidth=2, color=color, label='Average Merged Accuracy', markersize=6)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(37.5, 39.5)
    
    # Added lines together for a single legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=True, facecolor='white')
    
    plt.title('Performance Scaling with Calibration Size N', fontweight='semibold', pad=15)
    plt.tight_layout()
    plt.savefig('calibration_scaling_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Generated calibration_scaling_sweep.pdf")

if __name__ == '__main__':
    plot_robustness()
    plot_sweeps()
