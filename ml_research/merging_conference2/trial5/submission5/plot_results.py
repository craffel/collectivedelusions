import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    try:
        with open('sweep_results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("sweep_results.json not found. Run the sweeps script first.")
        return

    # Extract calibration sizes and methods
    N_list = sorted([int(k) for k in results.keys()])
    calibration_methods = ['None', 'SP-TAAC', 'TAAC', 'ZIO-CF', 'FDSA', 'JSSC']
    colors = {
        'None': '#9E9E9E',       # Grey
        'SP-TAAC': '#FF9800',     # Orange
        'TAAC': '#2196F3',        # Blue
        'ZIO-CF': '#03A9F4',      # Light Blue (dashed)
        'FDSA': '#E91E63',        # Pink
        'JSSC': '#4CAF50'         # Green (Proposed)
    }

    # 1. Plot Average Accuracy vs Lambda (at N=64)
    # We find the results for N=64, TA merge mode across lambdas
    N_target = 64
    lambdas = sorted([float(l) for l in results[str(N_target)]['TA'].keys()])
    
    plt.figure(figsize=(7, 5))
    for method in calibration_methods:
        accs = []
        for lam in lambdas:
            acc = results[str(N_target)]['TA'][str(lam)][method]['average']
            accs.append(acc)
        
        linestyle = '--' if method == 'ZIO-CF' else '-'
        marker = 'o' if method == 'JSSC' else 's' if method == 'FDSA' else '^' if method == 'TAAC' else 'x'
        plt.plot(lambdas, accs, label=method, color=colors[method], linestyle=linestyle, marker=marker, linewidth=2)
        
    plt.title(f"Joint Spatial-Spectral Calibration Performance (N={N_target})", fontsize=12, fontweight='bold')
    plt.xlabel("Task Arithmetic Scaling Coefficient ($\lambda$)", fontsize=11)
    plt.ylabel("Average Test Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig('accuracy_vs_lambda.png', dpi=300)
    plt.close()
    print("Saved accuracy_vs_lambda.png")

    # 2. Plot Average Accuracy vs Calibration Size N (at lambda=0.3)
    lam_target = 0.3
    plt.figure(figsize=(7, 5))
    for method in calibration_methods:
        accs = []
        for N in N_list:
            acc = results[str(N)]['TA'][str(lam_target)][method]['average']
            accs.append(acc)
            
        linestyle = '--' if method == 'ZIO-CF' else '-'
        marker = 'o' if method == 'JSSC' else 's' if method == 'FDSA' else '^' if method == 'TAAC' else 'x'
        plt.plot(N_list, accs, label=method, color=colors[method], linestyle=linestyle, marker=marker, linewidth=2)
        
    plt.title(f"Sample Efficiency of Calibration Methods ($\lambda$={lam_target})", fontsize=12, fontweight='bold')
    plt.xlabel("Calibration Dataset Size ($N$ per task)", fontsize=11)
    plt.ylabel("Average Test Accuracy (%)", fontsize=11)
    plt.xscale('log')
    plt.xticks(N_list, N_list)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig('accuracy_vs_N.png', dpi=300)
    plt.close()
    print("Saved accuracy_vs_N.png")

    # 3. Plot Task-specific Accuracy for WA and TA (N=64, lambda=0.3)
    # Grouped Bar chart
    tasks = ['mnist', 'fashion_mnist', 'cifar10']
    x = np.arange(len(tasks))
    width = 0.12

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # We plot WA
    methods_to_plot = ['None', 'TAAC', 'FDSA', 'JSSC']
    for i, method in enumerate(methods_to_plot):
        accs = [results['64']['WA'][method]['tasks'][t] for t in tasks]
        offset = (i - len(methods_to_plot)/2 + 0.5) * width
        rects = ax.bar(x + offset, accs, width, label=f"WA + {method}", color=colors[method], alpha=0.85)

    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Task-Specific Accuracy Comparison (WA, N=64)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['MNIST', 'Fashion-MNIST', 'CIFAR-10'], fontsize=10)
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig('task_comparison_wa.png', dpi=300)
    plt.close()
    print("Saved task_comparison_wa.png")

if __name__ == '__main__':
    generate_plots()
