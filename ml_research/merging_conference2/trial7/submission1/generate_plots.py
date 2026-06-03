import json
import numpy as np
import matplotlib.pyplot as plt

def generate_plots():
    seeds = [42, 43, 44]
    scenarios = ["A", "B", "C", "D", "E"]
    sc_names = {
        "A": "SGD Std (1e-4)",
        "B": "SGD High Decay (1e-2)",
        "C": "AdamW Std (1e-4)",
        "D": "AdamW High LR (1e-3)",
        "E": "AdamW High Decay (1e-2)"
    }
    methods = ["uncal", "bnc", "sptaac", "hybrid4", "hybrid8"]
    method_labels = ["Uncalibrated", "BNC", "SP-TAAC", "Hybrid (Rank 4)", "Hybrid (Rank 8)"]
    
    data = {sc: {m: [] for m in methods} for sc in scenarios}
    
    for s in seeds:
        with open(f"results_seed{s}.json", "r") as f:
            res = json.load(f)
            for sc in scenarios:
                data[sc]["uncal"].append(res[sc]["avg_uncal"])
                data[sc]["bnc"].append(res[sc]["avg_bnc"])
                data[sc]["sptaac"].append(res[sc]["avg_sptaac"])
                data[sc]["hybrid4"].append(res[sc]["avg_hybrid4"])
                data[sc]["hybrid8"].append(res[sc]["avg_hybrid8"])
                
    # Compute mean and standard deviation
    means = {sc: [np.mean(data[sc][m]) for m in methods] for sc in scenarios}
    stds = {sc: [np.std(data[sc][m]) for m in methods] for sc in scenarios}
    
    # Set up the style of the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    markers = ['o', 's', '^', 'D', 'x']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for i, sc in enumerate(scenarios):
        ax.errorbar(
            method_labels, 
            means[sc], 
            yerr=stds[sc], 
            label=sc_names[sc],
            marker=markers[i],
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=2,
            markersize=8,
            capsize=5,
            elinewidth=1.5
        )
        
    ax.set_ylabel("Average Test Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Calibration Method", fontsize=12, fontweight='bold')
    ax.set_title("Multi-Task Merged Model Accuracy by Calibration Method", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Place legend inside or outside
    ax.legend(frameon=True, facecolor='white', edgecolor='lightgray', fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig("calibration_results.pdf", dpi=300)
    plt.savefig("calibration_results.png", dpi=300)
    print("Plots generated successfully and saved as calibration_results.pdf and calibration_results.png!")

if __name__ == "__main__":
    generate_plots()
