import json
import matplotlib.pyplot as plt
import numpy as np

# Load experiment results
try:
    with open("experiment_results.json", "r") as f:
        results = json.load(f)
except Exception as e:
    print(f"Error loading experiment_results.json: {e}")
    # Fallback to dummy data for development if file doesn't exist yet
    results = {}

if results:
    stream_types = ['alternating', 'sequential']
    corruptions = ['none', 'gaussian_noise', 'gaussian_blur', 'contrast']
    corr_labels = ['Clean', 'G-Noise', 'G-Blur', 'Contrast']
    
    methods = ['uniform', 'adamerging', 'lfwa', 'pc_merge', 'cpa_merge', 'ewfr_merge']
    method_labels = ['Uniform', 'AdaMerging', 'LFWA', 'PC-Merge', 'CPA-Merge', 'EWFR-Merge (Ours)']
    colors = ['#7f7f7f', '#aec7e8', '#1f77b4', '#ffbb78', '#ff7f0e', '#d62728']
    
    # Let's create two subplots: one for Alternating, one for Sequential
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    
    for idx, stream_type in enumerate(stream_types):
        ax = axes[idx]
        
        # Extract data
        data_matrix = []
        for method in methods:
            method_accs = []
            for corr in corruptions:
                acc = results.get(stream_type, {}).get(corr, {}).get(method, 0.0)
                method_accs.append(acc)
            data_matrix.append(method_accs)
            
        # Plotting bars
        x = np.arange(len(corruptions))
        width = 0.13
        
        for m_idx, (method_data, label, color) in enumerate(zip(data_matrix, method_labels, colors)):
            offset = (m_idx - len(methods)/2) * width + width/2
            rects = ax.bar(x + offset, method_data, width, label=label, color=color, edgecolor='black', linewidth=0.5)
            
        ax.set_title(f"{stream_type.upper()} Stream Adaptation", fontsize=14, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(corr_labels, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
            
    # Add common legend below the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05), fontsize=11, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save high-resolution figure for publication
    plt.savefig("results_plot.png", dpi=300, bbox_inches='tight')
    print("Successfully generated and saved results_plot.png")
else:
    print("No results to plot.")
