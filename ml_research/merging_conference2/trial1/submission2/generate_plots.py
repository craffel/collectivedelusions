import json
import matplotlib.pyplot as plt
import numpy as np

# Set professional plotting style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'text.usetex': False  # Disable LaTeX rendering to prevent compilation issues
})

def plot_main_results():
    with open('results.json', 'r') as f:
        data = json.load(f)
        
    scalings = [item['scaling'] for item in data['task_arithmetic']]
    
    ta_acc = [item['average'] for item in data['task_arithmetic']]
    iso_acc = [item['average'] for item in data['isotropic']]
    wsa_acc = [item['average'] for item in data['wsa']]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(scalings, ta_acc, 'o-', color='#1f77b4', linewidth=2, label='Task Arithmetic (TA)', markersize=6)
    ax.plot(scalings, iso_acc, 's--', color='#2ca02c', linewidth=2, label='Isotropic Merging (Iso)', markersize=6)
    ax.plot(scalings, wsa_acc, '*-', color='#d62728', linewidth=2, label='WSA (Ours)', markersize=8)
    
    ax.set_xlabel('Scaling Factor ($\lambda_{\mathrm{scale}}$)')
    ax.set_ylabel('Average Multi-task Test Accuracy (%)')
    ax.set_title('Model Merging Performance Sweep')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.savefig('results_plot.png', dpi=300)
    plt.close()
    print("Main results plot generated and saved as results_plot.png")

def plot_ablation_epsilon():
    with open('ablation_results.json', 'r') as f:
        data = json.load(f)
        
    epsilons_str = ['1e-8', '1e-6', '1e-4', '1e-2']
    epsilons = [1e-8, 1e-6, 1e-4, 1e-2]
    
    accuracies = [
        data['1e-08']['average_accuracy'],
        data['1e-06']['average_accuracy'],
        data['0.0001']['average_accuracy'],
        data['0.01']['average_accuracy']
    ]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Bar plot for epsilon ablation
    bars = ax.bar(epsilons_str, accuracies, color=['#9467bd', '#bcbd22', '#17becf', '#ff7f0e'], width=0.5, edgecolor='black', alpha=0.8)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    ax.set_xlabel('Regularization Parameter ($\epsilon$)')
    ax.set_ylabel('Average Multi-task Accuracy (%)')
    ax.set_title('Ablation Study of WSA Regularization')
    ax.set_ylim(0, max(accuracies) + 10)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('ablation_epsilon.png', dpi=300)
    plt.close()
    print("Ablation study plot generated and saved as ablation_epsilon.png")

if __name__ == '__main__':
    plot_main_results()
    plot_ablation_epsilon()
