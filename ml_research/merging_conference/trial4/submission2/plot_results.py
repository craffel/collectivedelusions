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
        
    corruptions = ['clean', 'gaussian_noise', 'gaussian_blur', 'contrast']
    methods = [
        'static_merged',
        'standard_tta_tf',
        'standard_tta_tg',
        's2c_merge',
        'ewc_tta',
        'uewc_merge'
    ]
    
    # Capitalize names for plotting
    corr_labels = ['Clean', 'Gaussian Noise', 'Gaussian Blur', 'Contrast']
    method_labels = {
        'static_merged': 'Static Merged',
        'standard_tta_tf': 'Standard TTA (Teacher-Free)',
        'standard_tta_tg': 'Standard TTA (Teacher-Guided)',
        's2c_merge': 'S2C-Merge (Teacher-Free)',
        'ewc_tta': 'EWC-TTA (Teacher-Guided)',
        'uewc_merge': 'UEWC-Merge (Teacher-Free - Ours)'
    }
    
    # Extract data
    data = {m: [] for m in methods}
    for corr in corruptions:
        for m in methods:
            data[m].append(results[corr][m])
            
    # Set up plot
    x = np.arange(len(corruptions))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = {
        'static_merged': '#7f8c8d',
        'standard_tta_tf': '#e74c3c',
        'standard_tta_tg': '#e67e22',
        's2c_merge': '#3498db',
        'ewc_tta': '#9b59b6',
        'uewc_merge': '#2ecc71'
    }
    
    for idx, m in enumerate(methods):
        ax.bar(x + (idx - 2.5) * width, data[m], width, label=method_labels[m], color=colors[m])
        
    ax.set_ylabel('Multi-Task Average Accuracy (%)', fontsize=12)
    ax.set_title('Test-Time Model Merging Performance under Domain Shifts', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(corr_labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    
    # Annotate values on top of the bars for Ours and S2C-Merge
    for i in range(len(corruptions)):
        # Ours (UEWC-Merge)
        ours_val = results[corruptions[i]]['uewc_merge']
        ax.text(i + 2.5*width, ours_val + 1, f"{ours_val:.1f}%", ha='center', va='bottom', fontsize=8, color='#27ae60', fontweight='bold')
        
        # S2C-Merge
        s2c_val = results[corruptions[i]]['s2c_merge']
        ax.text(i + 0.5*width, s2c_val + 1, f"{s2c_val:.1f}%", ha='center', va='bottom', fontsize=8, color='#2980b9')
        
    plt.tight_layout()
    plot_path = './experts/results_plot.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Successfully generated results plot and saved to {plot_path}")

if __name__ == '__main__':
    generate_plot()
