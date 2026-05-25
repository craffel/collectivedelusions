import numpy as np
import matplotlib.pyplot as plt
import os

def generate_reports_and_plots():
    if not os.path.exists('evaluation_results.npz'):
        print("No evaluation results found! Please wait for eval_ttmm.py to complete.")
        return
        
    data = np.load('evaluation_results.npz', allow_pickle=True)
    results = data['results'].item()
    
    streams = ['alternating', 'sequential']
    corruptions = ['clean', 'noise', 'blur', 'contrast']
    methods = ['static', 'adamerging', 'lfwa', 'pc-merge', 'iggs-merge']
    
    method_labels = {
        'static': 'Static Merging',
        'adamerging': 'AdaMerging',
        'lfwa': 'LFWA',
        'pc-merge': 'PC-Merge + OPR',
        'iggs-merge': 'IGGS-Merge + OPR (Ours)'
    }
    
    # 1. Print Markdown Tables
    for stream in streams:
        print(f"\n### {stream.upper()} STREAM EVALUATION RESULTS")
        print("| Method | Clean | Gaussian Noise | Gaussian Blur | Contrast Shift | Average |")
        print("|---|---|---|---|---|---|")
        
        for method in methods:
            row = f"| **{method_labels[method]}** |"
            accs = []
            for corr in corruptions:
                acc = results[stream][corr][method]
                row += f" {acc:.2f}% |"
                accs.append(acc)
            mean_acc = np.mean(accs)
            row += f" **{mean_acc:.2f}%** |"
            print(row)
            
    # 2. Generate Plots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(corruptions))
    width = 0.15
    
    colors = {
        'static': '#95a5a6',
        'adamerging': '#3498db',
        'lfwa': '#e67e22',
        'pc-merge': '#9b59b6',
        'iggs-merge': '#e74c3c'
    }
    
    for idx, stream in enumerate(streams):
        ax = axes[idx]
        for m_idx, method in enumerate(methods):
            y = [results[stream][corr][method] for corr in corruptions]
            rects = ax.bar(x + (m_idx - 2) * width, y, width, label=method_labels[method], color=colors[method])
            
        ax.set_title(f"{stream.capitalize()} Stream Accuracy Comparison", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() if c != 'noise' else 'Gaussian Noise' for c in corruptions], fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_ylim(0, 105)
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=10)
            
    plt.tight_layout()
    plt.savefig('stream_comparison.png', dpi=300)
    print("\nSaved comparison plot to stream_comparison.png.")

if __name__ == '__main__':
    generate_reports_and_plots()
