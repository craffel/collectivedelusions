import json
import numpy as np
import matplotlib.pyplot as plt

# Set style for academic paper
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

def main():
    with open("results.json", "r") as f:
        results = json.load(f)
        
    seeds = list(results.keys())
    
    # Extract threshold sweep results for Pre-BN and Post-BN
    # Thresholds mapped:
    # LSC -> tau = 1.1
    # SMACS (tau=0.95) -> tau = 0.95
    # SMACS (tau=0.90) -> tau = 0.90
    # SMACS (tau=0.70) -> tau = 0.70
    # SMACS (tau=0.50) -> tau = 0.50
    # SMACS (tau=0.30) -> tau = 0.30
    # SMACS (tau=0.10) -> tau = 0.10
    # TCAC/SAC -> tau = -0.1
    
    tau_configs = [
        (-0.1, "TCAC/SAC"),
        (0.10, "SMACS (tau=0.10)"),
        (0.30, "SMACS (tau=0.30)"),
        (0.50, "SMACS (tau=0.50)"),
        (0.70, "SMACS (tau=0.70)"),
        (0.90, "SMACS (tau=0.90)"),
        (0.95, "SMACS (tau=0.95)"),
        (1.10, "LSC")
    ]
    
    pre_bn_means = []
    pre_bn_stds = []
    post_bn_means = []
    post_bn_stds = []
    
    for tau, name in tau_configs:
        # Pre-BN key matching
        if name == "LSC":
            pre_key = "Pre-BN LSC"
            post_key = "Post-BN LSC"
        elif name == "TCAC/SAC":
            pre_key = "Pre-BN TCAC/SAC"
            post_key = "Post-BN TCAC/SAC"
        else:
            pre_key = f"Pre-BN {name}"
            post_key = f"Post-BN {name}"
            
        # Collect across seeds
        pre_vals = [results[s][pre_key]["average"] for s in seeds]
        post_vals = [results[s][post_key]["average"] for s in seeds]
        
        pre_bn_means.append(np.mean(pre_vals))
        pre_bn_stds.append(np.std(pre_vals))
        post_bn_means.append(np.mean(post_vals))
        post_bn_stds.append(np.std(post_vals))
        
    taus = [t[0] for t in tau_configs]
    
    # 1. Generate Threshold Sweep Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    
    ax.errorbar(taus, pre_bn_means, yerr=pre_bn_stds, fmt='-o', color='#1f77b4', 
                capsize=5, elinewidth=1.5, markeredgewidth=1.5, label='Pre-BatchNorm (Input)')
    ax.fill_between(taus, np.array(pre_bn_means) - np.array(pre_bn_stds), 
                    np.array(pre_bn_means) + np.array(pre_bn_stds), color='#1f77b4', alpha=0.15)
    
    ax.errorbar(taus, post_bn_means, yerr=post_bn_stds, fmt='-s', color='#d62728', 
                capsize=5, elinewidth=1.5, markeredgewidth=1.5, label='Post-BatchNorm (Output)')
    ax.fill_between(taus, np.array(post_bn_means) - np.array(post_bn_stds), 
                    np.array(post_bn_means) + np.array(post_bn_stds), color='#d62728', alpha=0.15)
    
    # Add baseline
    uncal_vals = [results[s]["Uncalibrated"]["average"] for s in seeds]
    uncal_mean = np.mean(uncal_vals)
    ax.axhline(y=uncal_mean, color='#2ca02c', linestyle='--', linewidth=1.5, label='Uncalibrated Baseline')
    
    ax.set_xlabel('Sparsity Masking Threshold ($\\tau$)')
    ax.set_ylabel('Average Multi-Task Accuracy (%)')
    ax.set_title('Sparsity Threshold Sweep: Capacity vs. Numerical Stability')
    ax.set_xticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.1])
    ax.set_xticklabels(['-0.1\n(TCAC)', '0.1', '0.3', '0.5', '0.7', '0.9', '0.95', '1.1\n(LSC)'])
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('plot_threshold_sweep.pdf')
    plt.close()
    
    # 2. Generate Calibration-Adaptation Conflict Plot
    # Methods to compare:
    # 1. Uncalibrated (Baseline)
    # 2. Pre-BN SMACS (tau=0.50) [Best Backbone Calibration]
    # 3. Post-BN LSC [Best Post-BN Calibration]
    # 4. Head-only Adaptation
    # 5. Pre-BN SMACS (tau=0.50) + Head Adaptation
    # 6. Post-BN SMACS (tau=0.95) + Head Adaptation
    
    conflict_methods = [
        ("Uncalibrated", "Uncalibrated\nBaseline"),
        ("Pre-BN SMACS (tau=0.50)", "Pre-BN\nSMACS ($\\tau$=0.5)"),
        ("Post-BN LSC", "Post-BN\nLSC (Pure Layer)"),
        ("Pre-BN SMACS (tau=0.50) + Head Adaptation", "Pre-BN SMACS\n+ Head Adapt"),
        ("Post-BN SMACS (tau=0.95) + Head Adaptation", "Post-BN SMACS\n+ Head Adapt"),
        ("Head-only Adaptation", "Head-only\nAdaptation")
    ]
    
    labels = []
    means = []
    stds = []
    colors = ['#7f7f7f', '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']
    
    for key, label in conflict_methods:
        vals = [results[s][key]["average"] for s in seeds]
        labels.append(label)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor='none', alpha=0.85, capsize=5, width=0.55)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')
                    
    ax.set_ylabel('Average Multi-Task Accuracy (%)')
    ax.set_title('The Calibration-Adaptation Conflict in Model Merging')
    ax.set_ylim(0, 80)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig('plot_calibration_adaptation.pdf')
    plt.close()
    print("Plots generated successfully!")

if __name__ == "__main__":
    main()
