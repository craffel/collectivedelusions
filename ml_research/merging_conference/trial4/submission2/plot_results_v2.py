import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_plot():
    results_path = './experts/evaluation_results.json'
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found. Running plot with dummy/template data...")
        return
        
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    corruptions = ['clean', 'gaussian_noise', 'gaussian_blur', 'contrast', 'brightness']
    methods = [
        'static_merged',
        'standard_tta_tf',
        'standard_tta_tg',
        's2c_merge',
        'ewc_tta',
        'uewc_merge_no_ewc',
        'uewc_merge'
    ]
    
    # Capitalize names for plotting
    corr_labels = ['Clean', 'Gaussian Noise', 'Gaussian Blur', 'Contrast', 'Brightness']
    method_labels = {
        'static_merged': 'Static Merged',
        'standard_tta_tf': 'Standard TTA (Teacher-Free)',
        'standard_tta_tg': 'Standard TTA (Teacher-Guided)',
        's2c_merge': 'S2C-Merge (Teacher-Free)',
        'ewc_tta': 'EWC-TTA (Teacher-Guided)',
        'uewc_merge_no_ewc': 'UEWC-Merge (Ours - no EWC)',
        'uewc_merge': 'UEWC-Merge (Ours)'
    }
    
    # Extract data
    means = {m: [] for m in methods}
    stds = {m: [] for m in methods}
    for corr in corruptions:
        for m in methods:
            means[m].append(results[corr][m]['mean'])
            stds[m].append(results[corr][m]['std'])
            
    # Set up plot
    x = np.arange(len(corruptions))
    width = 0.10
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    colors = {
        'static_merged': '#7f8c8d',
        'standard_tta_tf': '#e74c3c',
        'standard_tta_tg': '#e67e22',
        's2c_merge': '#3498db',
        'ewc_tta': '#9b59b6',
        'uewc_merge_no_ewc': '#1abc9c',
        'uewc_merge': '#2ecc71'
    }
    
    for idx, m in enumerate(methods):
        ax.bar(
            x + (idx - 3.0) * width, 
            means[m], 
            width, 
            yerr=stds[m], 
            capsize=2, 
            label=method_labels[m], 
            color=colors[m],
            error_kw={'elinewidth': 1, 'ecolor': '#333333'}
        )
        
    ax.set_ylabel('Multi-Task Average Accuracy (%)', fontsize=12)
    ax.set_title('Test-Time Model Merging Performance under Domain Shifts (5 Seeds)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(corr_labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9, ncol=2)
    
    # Annotate values on top of the bars for Ours (UEWC-Merge)
    for i in range(len(corruptions)):
        # Ours (UEWC-Merge)
        ours_val = results[corruptions[i]]['uewc_merge']['mean']
        ax.text(i + 3.0*width, ours_val + 2, f"{ours_val:.1f}%", ha='center', va='bottom', fontsize=8, color='#27ae60', fontweight='bold')
        
    plt.tight_layout()
    plot_path = './experts/results_plot.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Successfully generated results plot and saved to {plot_path}")

if __name__ == '__main__':
    generate_plot()
