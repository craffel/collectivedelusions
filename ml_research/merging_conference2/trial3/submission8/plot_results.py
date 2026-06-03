import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_main_benchmark(data):
    # Data is a dict of {setting: {method: {task: acc}}}
    # Setting: WA, TA (λ=0.1), TA (λ=0.2), TA (λ=0.3), TA (λ=0.4)
    # Methods: none, tcac, taac, lsc, sp_taac
    settings = list(data.keys())
    methods = ['none', 'tcac', 'taac', 'lsc', 'sp_taac']
    labels = {
        'none': 'Uncalibrated',
        'tcac': 'TCAC (Task-Cond)',
        'taac': 'TAAC (Task-Agn)',
        'lsc': 'LSC (Task-Cond)',
        'sp_taac': 'SP-TAAC (Ours, Task-Agn)'
    }
    colors = {
        'none': '#7f8c8d',
        'tcac': '#e74c3c',
        'taac': '#e67e22',
        'lsc': '#3498db',
        'sp_taac': '#2ecc71'
    }
    
    x = np.arange(len(settings))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, m in enumerate(methods):
        accs = []
        for s in settings:
            accs.append(data[s].get(m, {}).get('avg', 0.0))
        ax.bar(x + (i - 2) * width, accs, width, label=labels[m], color=colors[m])
        
    ax.set_ylabel('Average Test Accuracy (%)', fontsize=12)
    ax.set_xlabel('Merging Configuration', fontsize=12)
    ax.set_title('Multi-Task Model Merging Calibration Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, loc='lower left')
    
    plt.tight_layout()
    plt.savefig('results/main_benchmark.png', dpi=300)
    plt.close()
    print("Saved main benchmark plot to results/main_benchmark.png")

def plot_sample_efficiency(data):
    # Data is a dict of {mode: {N: {method: {task: acc}}}}
    # We plot WA sample efficiency (N-sweep)
    wa_data = data.get('wa', {})
    if not wa_data:
        print("No WA sample efficiency data to plot.")
        return
        
    ns = sorted([int(k) for k in wa_data.keys()])
    methods = ['none', 'tcac', 'taac', 'lsc', 'sp_taac']
    labels = {
        'none': 'Uncalibrated',
        'tcac': 'TCAC (Task-Cond)',
        'taac': 'TAAC (Task-Agn)',
        'lsc': 'LSC (Task-Cond)',
        'sp_taac': 'SP-TAAC (Ours, Task-Agn)'
    }
    colors = {
        'none': '#7f8c8d',
        'tcac': '#e74c3c',
        'taac': '#e67e22',
        'lsc': '#3498db',
        'sp_taac': '#2ecc71'
    }
    markers = {
        'none': 'x',
        'tcac': 's',
        'taac': 'o',
        'lsc': '^',
        'sp_taac': 'D'
    }
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    for m in methods:
        accs = []
        for n in ns:
            accs.append(wa_data[str(n)].get(m, {}).get('avg', 0.0))
        ax.plot(ns, accs, label=labels[m], color=colors[m], marker=markers[m], linewidth=2, markersize=8)
        
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.set_xlabel('Calibration Samples per Task ($N$)', fontsize=12)
    ax.set_ylabel('Average Test Accuracy (%)', fontsize=12)
    ax.set_title('Calibration Sample Efficiency under Weight Averaging', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('results/sample_efficiency.png', dpi=300)
    plt.close()
    print("Saved sample efficiency plot to results/sample_efficiency.png")

def plot_distributional_robustness():
    # We dynamically load task imbalance results and plot standard N=64 vs imbalanced MNIST
    # We compare TAAC and SP-TAAC (our method) to show that SP-TAAC is stable under imbalance
    tasks = ['mnist', 'fashion', 'cifar']
    methods = ['taac', 'sp_taac']
    labels = {'taac': 'TAAC (Channel-wise)', 'sp_taac': 'SP-TAAC (Ours, Layer-wise)'}
    colors = {'taac': '#e67e22', 'sp_taac': '#2ecc71'}
    
    # We extract average accuracies for standard N=64 and imbalanced MNIST ratio 0.25 and ratio 4.0
    # under Weight Averaging
    x_labels = ['MNIST-Sparse (0.25x)', 'Balanced (1.0x)', 'MNIST-Heavy (4.0x)']
    ratios = [0.25, 1.0, 4.0]
    
    taac_accs = []
    sp_taac_accs = []
    
    # Ratio 0.25
    try:
        with open("results/imbalance/wa_task_mnist_ratio0.25.json", "r") as f:
            res = json.load(f)
            taac_accs.append(res['taac']['avg'])
            sp_taac_accs.append(res['sp_taac']['avg'])
    except Exception:
        taac_accs.append(0.0)
        sp_taac_accs.append(0.0)
        
    # Ratio 1.0 (standard N=64)
    try:
        with open("results/sample_efficiency/wa_N64.json", "r") as f:
            res = json.load(f)
            taac_accs.append(res['taac']['avg'])
            sp_taac_accs.append(res['sp_taac']['avg'])
    except Exception:
        taac_accs.append(0.0)
        sp_taac_accs.append(0.0)
        
    # Ratio 4.0
    try:
        with open("results/imbalance/wa_task_mnist_ratio4.0.json", "r") as f:
            res = json.load(f)
            taac_accs.append(res['taac']['avg'])
            sp_taac_accs.append(res['sp_taac']['avg'])
    except Exception:
        taac_accs.append(0.0)
        sp_taac_accs.append(0.0)
        
    fig, ax = plt.subplots(figsize=(7, 5))
    
    x = np.arange(len(x_labels))
    ax.plot(x, taac_accs, label=labels['taac'], color=colors['taac'], marker='o', linewidth=2.5, markersize=8)
    ax.plot(x, sp_taac_accs, label=labels['sp_taac'], color=colors['sp_taac'], marker='D', linewidth=2.5, markersize=8)
    
    ax.set_ylabel('Average Test Accuracy (%)', fontsize=12)
    ax.set_xlabel('Calibration Set Distribution (under Weight Averaging)', fontsize=12)
    ax.set_title('Distributional Robustness under Task Imbalance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='lower left')
    
    plt.tight_layout()
    plt.savefig('results/distributional_robustness.png', dpi=300)
    plt.close()
    print("Saved distributional robustness plot to results/distributional_robustness.png")

def main():
    if not os.path.exists('results/compiled_summary.json'):
        print("No compiled summary found. Please run run_all_sweeps.py first.")
        return
        
    with open('results/compiled_summary.json', 'r') as f:
        summary = json.load(f)
        
    plot_main_benchmark(summary.get('main_benchmark', {}))
    plot_sample_efficiency(summary.get('sample_efficiency', {}))
    plot_distributional_robustness()

if __name__ == '__main__':
    main()
