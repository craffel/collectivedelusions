import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    metrics_path = 'results/metrics.json'
    if not os.path.exists(metrics_path):
        print("ERROR: metrics.json does not exist. Please run run_regcalmerge.py first!")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    methods = {
        'Task Arithmetic': 'task_arithmetic',
        'AdaMerging (Adam)': 'adam_opt',
        'AdaMerging (ES)': 'es_opt',
        'Spatial Mean': 'spatial_mean_es',
        'RegCalMerge (Ours)': 'regcalmerge_opt'
    }

    # Extract means and stds
    data = {m: [] for m in methods}
    errors = {m: [] for m in methods}

    for method_name, method_key in methods.items():
        for ds in datasets:
            mean = metrics[method_key][ds]['mean'] * 100
            std = metrics[method_key][ds]['std'] * 100
            data[method_name].append(mean)
            errors[method_name].append(std)

    # Set up plot
    x = np.arange(len(datasets))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = ['#cccccc', '#888888', '#555555', '#3366cc', '#ff9900']

    for i, (method_name, values) in enumerate(data.items()):
        ax.bar(x + (i - 2) * width, values, width, label=method_name, 
               yerr=errors[method_name], capsize=3, color=colors[i], edgecolor='black', alpha=0.9)

    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test-Time Model Merging Performance on CLIP ViT-B/32', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
    ax.set_ylim(40, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=10)

    # Display average performance on the plot
    text_y = 95
    for i, (method_name, values) in enumerate(data.items()):
        avg = np.mean(values)
        ax.text(x[-1] + 0.8, text_y, f"{method_name}: {avg:.2f}%", 
                fontsize=9, color=colors[i], fontweight='bold')
        text_y -= 4

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plot_path = 'results/fig1.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot successfully saved to {plot_path}")

if __name__ == '__main__':
    main()
