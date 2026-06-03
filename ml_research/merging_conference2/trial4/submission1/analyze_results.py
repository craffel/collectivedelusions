import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze():
    try:
        data = torch.load("results_sweep.pt")
    except Exception as e:
        print(f"Error loading results_sweep.pt: {e}")
        return

    results = data["results"]
    target_stds = data["target_stds"]

    print("\n================== SWEEP ANALYSIS ==================")
    print(f"Target activation stds across layers: {target_stds}")
    
    # Organize by method
    methods = ["WA", "TA", "TIES", "DARE"]
    
    for method in methods:
        method_runs = [r for r in results if r["method"] == method]
        if not method_runs:
            continue
            
        print(f"\n--- Analysis for {method} ---")
        
        # Uncalibrated runs
        uncal_runs = [r for r in method_runs if not r["calibrated"]]
        # Calibrated runs
        cal_runs = [r for r in method_runs if r["calibrated"]]
        
        if uncal_runs:
            best_uncal = max(uncal_runs, key=lambda x: x["mean_acc"])
            print(f"Best UNCALIBRATED: {best_uncal['params']} | Mean Acc = {best_uncal['mean_acc']:.2f}% | Accs: MNIST={best_uncal['accs']['mnist']:.2f}%, Fashion={best_uncal['accs']['fashion']:.2f}%, CIFAR={best_uncal['accs']['cifar']:.2f}%")
            print(f"  Stds: {best_uncal['stds']}")
            
        if cal_runs:
            best_cal = max(cal_runs, key=lambda x: x["mean_acc"])
            print(f"Best CALIBRATED: {best_cal['params']} | Mean Acc = {best_cal['mean_acc']:.2f}% | Accs: MNIST={best_cal['accs']['mnist']:.2f}%, Fashion={best_cal['accs']['fashion']:.2f}%, CIFAR={best_cal['accs']['cifar']:.2f}%")
            print(f"  Stds: {best_cal['stds']}")
            
        # Comparison
        if uncal_runs and cal_runs:
            diff = best_uncal['mean_acc'] - best_cal['mean_acc']
            print(f"Difference (Uncal - Cal): {diff:+.2f}%")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 1. TA scale sweep
    ta_runs = [r for r in results if r["method"] == "TA"]
    if ta_runs:
        scales = []
        uncal_accs = []
        cal_accs = []
        for r in ta_runs:
            scale = float(r["params"].split("=")[1])
            if r["calibrated"]:
                cal_accs.append((scale, r["mean_acc"]))
            else:
                uncal_accs.append((scale, r["mean_acc"]))
        
        uncal_accs.sort()
        cal_accs.sort()
        
        ax = axes[0]
        ax.plot([x[0] for x in uncal_accs], [x[1] for x in uncal_accs], 'o-', label='Uncalibrated TA')
        ax.plot([x[0] for x in cal_accs], [x[1] for x in cal_accs], 's--', label='Calibrated TA (SP-TAAC)')
        ax.set_xlabel('Scale')
        ax.set_ylabel('Mean Accuracy (%)')
        ax.set_title('Task Arithmetic Scale Sweep')
        ax.legend()
        ax.grid(True)

    # 2. TIES scale sweep (at best keep_rate)
    ties_runs = [r for r in results if r["method"] == "TIES"]
    if ties_runs:
        # Find best keep_rate
        best_kr = 0.2
        uncal_ties = [r for r in ties_runs if not r["calibrated"]]
        if uncal_ties:
            best_run = max(uncal_ties, key=lambda x: x["mean_acc"])
            # extract kr
            parts = best_run["params"].split(",")
            best_kr = float(parts[0].split("=")[1])
            
        print(f"\nPlotting TIES sweep at best keep_rate={best_kr}")
        
        scales = []
        uncal_accs = []
        cal_accs = []
        for r in ties_runs:
            parts = r["params"].split(",")
            kr = float(parts[0].split("=")[1])
            scale = float(parts[1].split("=")[1])
            if kr == best_kr:
                if r["calibrated"]:
                    cal_accs.append((scale, r["mean_acc"]))
                else:
                    uncal_accs.append((scale, r["mean_acc"]))
                    
        uncal_accs.sort()
        cal_accs.sort()
        
        ax = axes[1]
        ax.plot([x[0] for x in uncal_accs], [x[1] for x in uncal_accs], 'o-', label=f'Uncal TIES (kr={best_kr})')
        ax.plot([x[0] for x in cal_accs], [x[1] for x in cal_accs], 's--', label=f'Cal TIES (kr={best_kr})')
        ax.set_xlabel('Scale')
        ax.set_ylabel('Mean Accuracy (%)')
        ax.set_title(f'TIES-Merging Scale Sweep (keep_rate={best_kr})')
        ax.legend()
        ax.grid(True)

    # 3. DARE scale sweep (at best drop_rate)
    dare_runs = [r for r in results if r["method"] == "DARE"]
    if dare_runs:
        best_dr = 0.9
        uncal_dare = [r for r in dare_runs if not r["calibrated"]]
        if uncal_dare:
            best_run = max(uncal_dare, key=lambda x: x["mean_acc"])
            parts = best_run["params"].split(",")
            best_dr = float(parts[0].split("=")[1])
            
        print(f"Plotting DARE sweep at best drop_rate={best_dr}")
        
        uncal_accs = []
        cal_accs = []
        for r in dare_runs:
            parts = r["params"].split(",")
            dr = float(parts[0].split("=")[1])
            scale = float(parts[1].split("=")[1])
            if dr == best_dr:
                if r["calibrated"]:
                    cal_accs.append((scale, r["mean_acc"]))
                else:
                    uncal_accs.append((scale, r["mean_acc"]))
                    
        uncal_accs.sort()
        cal_accs.sort()
        
        ax = axes[2]
        ax.plot([x[0] for x in uncal_accs], [x[1] for x in uncal_accs], 'o-', label=f'Uncal DARE (dr={best_dr})')
        ax.plot([x[0] for x in cal_accs], [x[1] for x in cal_accs], 's--', label=f'Cal DARE (dr={best_dr})')
        ax.set_xlabel('Scale')
        ax.set_ylabel('Mean Accuracy (%)')
        ax.set_title(f'DARE Scale Sweep (drop_rate={best_dr})')
        ax.legend()
        ax.grid(True)

    # 4. Activation Std comparison for WA and TA (scale=1.0 vs best uncal scale vs target)
    ax = axes[3]
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    x = np.arange(len(layers))
    width = 0.2
    
    # Get target stds
    y_target = [target_stds[l] for l in layers]
    ax.bar(x - width*1.5, y_target, width, label='Target (Experts)', color='gray')
    
    # Get WA (uncalibrated, which is scale=0.33 basically, since average of 3)
    wa_run = [r for r in results if r["method"] == "WA" and not r["calibrated"]][0]
    y_wa = [wa_run["stds"][l] for l in layers]
    ax.bar(x - width*0.5, y_wa, width, label='WA (Uncalibrated)', color='red')
    
    # Get TA (scale=0.3)
    ta_03 = [r for r in results if r["method"] == "TA" and r["params"] == "scale=0.3" and not r["calibrated"]]
    if ta_03:
        y_ta_03 = [ta_03[0]["stds"][l] for l in layers]
        ax.bar(x + width*0.5, y_ta_03, width, label='TA (scale=0.3)', color='orange')
        
    # Get TA (best uncal scale)
    if ta_runs:
        uncal_ta_runs = [r for r in ta_runs if not r["calibrated"]]
        best_ta_run = max(uncal_ta_runs, key=lambda x: x["mean_acc"])
        best_scale = float(best_ta_run["params"].split("=")[1])
        y_best_ta = [best_ta_run["stds"][l] for l in layers]
        ax.bar(x + width*1.5, y_best_ta, width, label=f'TA (best scale={best_scale})', color='green')
        
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel('Activation Std')
    ax.set_title('Activation Std Comparison across Layers')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("sweep_analysis.png", dpi=300)
    print("Analysis plot saved as sweep_analysis.png")

if __name__ == "__main__":
    analyze()
