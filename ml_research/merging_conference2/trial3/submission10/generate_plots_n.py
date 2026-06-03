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
    try:
        with open("results_n_sweep.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("results_n_sweep.json not found yet.")
        return

    n_values = [32, 128, 512]
    seeds = ["42", "43", "44"]
    
    # Methods to track and plot
    methods_to_plot = [
        ("Uncalibrated", "Uncalibrated Baseline", "o", "#7f7f7f", "-"),
        ("Post-BN LSC", "Post-BN LSC (Pure Layer)", "s", "#aec7e8", "-"),
        ("Post-BN TCAC/SAC", "Post-BN TCAC (Pure Channel)", "x", "#d62728", "--"),
        ("Head-only Adaptation", "Head-only Adaptation", "^", "#2ca02c", "-"),
        ("Post-BN SMACS (tau=0.95)", "Post-BN SMACS ($\\tau$=0.95)", "d", "#1f77b4", "-"),
        ("Post-BN SMACS (tau=0.95) + Head Adaptation", "Post-BN SMACS + Head Adapt", "p", "#ffbb78", "-.")
    ]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    
    for key, label, marker, color, linestyle in methods_to_plot:
        means = []
        stds = []
        for n in n_values:
            n_str = str(n)
            vals = []
            for seed in seeds:
                if n_str in results and seed in results[n_str]:
                    # Find matching keys
                    found_val = None
                    for k in results[n_str][seed].keys():
                        if key in k:
                            found_val = results[n_str][seed][k]["average"]
                            break
                    if found_val is not None:
                        vals.append(found_val)
                    else:
                        # If exact key not found, try default/uncalibrated
                        if key == "Uncalibrated":
                            vals.append(results[n_str][seed]["Uncalibrated"]["average"])
            
            if len(vals) > 0:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                
        # Only plot if we have valid data
        if not np.isnan(means).all():
            ax.errorbar(n_values, means, yerr=stds, fmt=marker + linestyle, color=color,
                        capsize=5, elinewidth=1.5, markeredgewidth=1.5, markersize=8, label=label)
            ax.fill_between(n_values, np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds), color=color, alpha=0.1)
            
    ax.set_xscale('log')
    ax.set_xticks(n_values)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Calibration Set Size ($N$)')
    ax.set_ylabel('Average Multi-Task Accuracy (%)')
    ax.set_title('Scaling Analysis: Influence of Calibration Set Size $N$')
    ax.set_ylim(15, 80)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6, which="both")
    
    plt.tight_layout()
    plt.savefig('plot_n_scaling.pdf')
    plt.close()
    print("plot_n_scaling.pdf generated successfully!")

if __name__ == "__main__":
    main()
