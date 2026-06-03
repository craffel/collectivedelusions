import json
import matplotlib.pyplot as plt
import numpy as np

# Set professional plotting style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

def generate_sweep_plot():
    with open('extended_analysis.json', 'r') as f:
        data = json.load(f)
        
    sweep_results = data['sweep_results']
    
    # Sort scales numerically
    scales = sorted([float(s) for s in sweep_results['Task Arithmetic'].keys()])
    scale_strs = [f"{s:.1f}" for s in scales]
    
    methods = {
        'Task Arithmetic': {'label': 'Task Arithmetic', 'color': '#1f77b4', 'marker': 'o', 'ls': '-'},
        'TIES Merging': {'label': 'TIES Merging', 'color': '#ff7f0e', 'marker': 's', 'ls': '--'},
        'DARE (drop=0.2)': {'label': 'DARE (drop=0.2)', 'color': '#2ca02c', 'marker': '^', 'ls': '-.'},
        'DMC-Merge (Global)': {'label': 'DMC-Merge (Global)', 'color': '#d62728', 'marker': 'd', 'ls': ':'},
        'DMC-Merge (Conflict-Aware)': {'label': 'DMC-Merge (Conflict)', 'color': '#9467bd', 'marker': 'x', 'ls': '-'}
    }
    
    plt.figure(figsize=(7, 4.5))
    
    for m_key, m_info in methods.items():
        if m_key in sweep_results:
            accs = [sweep_results[m_key][s_str]['avg'] for s_str in scale_strs]
            plt.plot(scales, accs, label=m_info['label'], color=m_info['color'], 
                     marker=m_info['marker'], linestyle=m_info['ls'], linewidth=1.5, markersize=5)
            
    plt.xlabel('Scaling Factor ($\lambda$)')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Hyperparameter Scale Sweeps of Model Merging Methods')
    plt.xlim(0.05, 1.55)
    plt.xticks(np.arange(0.1, 1.6, 0.2))
    plt.ylim(10, 65)
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('plot_sweeps.png', dpi=300)
    plt.close()
    print("Saved plot_sweeps.png")

def generate_bottleneck_plot():
    with open('extended_analysis.json', 'r') as f:
        data = json.load(f)
        
    svd_analysis = data['svd_analysis']
    
    layers = list(svd_analysis.keys())
    cos_sims = [svd_analysis[l]['avg_cos_sim'] for l in layers]
    rel_diffs = [svd_analysis[l]['avg_rel_diff'] for l in layers]
    
    # We can clean up layer names for display on x-axis
    clean_layers = []
    for l in layers:
        name = l.replace('.weight', '')
        # E.g. 4.0.conv1 -> L4.0.C1
        name = name.replace('layer', 'L')
        name = name.replace('.conv', '.C')
        if name == '0':
            name = 'Conv1'
        clean_layers.append(name)
        
    x = np.arange(len(layers))
    
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    color = '#d62728'
    ax1.set_xlabel('ResNet-18 Convolutional Layers (Shallow to Deep)')
    ax1.set_ylabel('Cosine Similarity', color=color)
    ax1.plot(x, cos_sims, color=color, marker='o', linewidth=2, label='Cosine Similarity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.2, 0.8)
    
    ax2 = ax1.twinx()  
    color = '#1f77b4'
    ax2.set_ylabel('Relative Frobenius Norm Diff', color=color)
    ax2.plot(x, rel_diffs, color=color, marker='s', linewidth=2, linestyle='--', label='Relative Diff')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.6, 1.0)
    
    plt.title('SVD Projection Bottleneck Across ResNet-18 Layers')
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_layers, rotation=45, ha='right')
    
    fig.tight_layout()  
    plt.savefig('plot_bottleneck.png', dpi=300)
    plt.close()
    print("Saved plot_bottleneck.png")

if __name__ == '__main__':
    generate_sweep_plot()
    generate_bottleneck_plot()
