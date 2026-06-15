import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return
        
    files = [f for f in os.listdir(results_dir) if f.endswith(".json") and not f.startswith("plot")]
    if not files:
        print("No JSON results found.")
        return
        
    data = {}
    
    # Map optimizer names to nice labels
    opt_map = {
        "adamw": "AdamW (Baseline)",
        "sam": "SAM (Flatter Min)",
        "sabcd_literal": "SA-BCD (Literal)",
        "sabcd_standard_adam": "SA-BCD (Std Adam)",
        "sabcd_adam_gt": "SA-BCD (Adam GT)"
    }
    
    merg_map = {
        "task_arithmetic": "Task Arithmetic (Average)",
        "isotropic": "Isotropic Merging (SVD)",
        "spectral_dampening": "Spectral Dampening (Decay)"
    }
    
    # Structure: data[optimizer][merging] = (acc, bwt)
    optimizers_list = list(opt_map.values())
    merging_list = list(merg_map.values())
    
    # Initialize dictionary
    for opt_nice in optimizers_list:
        data[opt_nice] = {}
        for merg_nice in merging_list:
            data[opt_nice][merg_nice] = (0.0, 0.0)

    for file in files:
        path = os.path.join(results_dir, file)
        try:
            with open(path, "r") as f:
                res = json.load(f)
                
            opt = res.get("optimizer", "unknown")
            merg = res.get("merging", "unknown")
            acc = res.get("acc", 0.0)
            bwt = res.get("bwt", 0.0)
            
            opt_nice = opt_map.get(opt, opt)
            merg_nice = merg_map.get(merg, merg)
            
            if opt_nice in data and merg_nice in data[opt_nice]:
                data[opt_nice][merg_nice] = (acc, bwt)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({'font.size': 11})
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data for plotting
    x = np.arange(len(optimizers_list))
    width = 0.25
    
    # Group colors
    colors_acc = ["#1f77b4", "#aec7e8", "#ff7f0e"] # Blue-ish and Orange
    colors_bwt = ["#d62728", "#ff9896", "#9467bd"] # Red-ish and Purple
    
    # Plot Accuracy
    for idx, merg_nice in enumerate(merging_list):
        accs = [data[opt_nice][merg_nice][0] for opt_nice in optimizers_list]
        rects = axes[0].bar(x + (idx - 1) * width, accs, width, label=merg_nice, color=colors_acc[idx], edgecolor='black', alpha=0.85)
        
        # Add labels on top of bars
        for rect in rects:
            height = rect.get_height()
            if height > 10:  # Only label successful runs
                axes[0].annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
                            
    axes[0].set_title("Average Accuracy (ACC) % (Higher is Better)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(optimizers_list, rotation=30, ha="right")
    axes[0].set_ylim(0, 80)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Forgetting (BWT)
    for idx, merg_nice in enumerate(merging_list):
        bwts = [data[opt_nice][merg_nice][1] for opt_nice in optimizers_list]
        rects = axes[1].bar(x + (idx - 1) * width, bwts, width, label=merg_nice, color=colors_bwt[idx], edgecolor='black', alpha=0.85)
        
        # Add labels below/above bars
        for rect in rects:
            height = rect.get_height()
            if abs(height) > 0.01:
                axes[1].annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -10 if height < 0 else 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
                            
    axes[1].set_title("Backward Transfer (BWT) % (Less Negative is Better)")
    axes[1].set_ylabel("Forgetting (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(optimizers_list, rotation=30, ha="right")
    axes[1].set_ylim(-80, 5)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.suptitle("Deconstructing SAIM: Optimizer vs. Merging Dissection on Split CIFAR-100", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = "results/comparison_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Successfully saved results comparison plot to {plot_path}")

if __name__ == "__main__":
    main()
