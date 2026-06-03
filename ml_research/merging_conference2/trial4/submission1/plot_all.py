import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_sweeps(data):
    results = data["results_record"]
    target_stds = data["target_stds"]
    final_full = data["final_full_results"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. TA and LWS-TA
    ta_runs = [r for r in results if r["method"] == "TA"]
    ta_scales = sorted(list(set([float(r["params"].split("=")[1]) for r in ta_runs])))
    
    uncal_ta = []
    cal_ta = []
    for scale in ta_scales:
        uncal_run = [r for r in ta_runs if not r["calibrated"] and float(r["params"].split("=")[1]) == scale][0]
        cal_run = [r for r in ta_runs if r["calibrated"] and float(r["params"].split("=")[1]) == scale][0]
        uncal_ta.append(uncal_run["mean_acc"])
        cal_ta.append(cal_run["mean_acc"])
        
    axes[0].plot(ta_scales, uncal_ta, 'o-', color='C0', label='Uncalibrated TA')
    axes[0].plot(ta_scales, cal_ta, 's--', color='C1', label='Calibrated TA (SP-TAAC)')
    
    # Add LWS points
    lws_runs = [r for r in results if r["method"] == "LWS-TA"]
    for i, r in enumerate(lws_runs):
        marker = '^' if 'Focused' in r["params"] else 'v'
        axes[0].plot(0.5, r["mean_acc"], marker, color='red', markersize=10, 
                     label=f"LWS-TA ({r['params'].split(' ')[0]})")
                     
    axes[0].set_xlabel('Scale Factor $\lambda$')
    axes[0].set_ylabel('Mean Accuracy (%)')
    axes[0].set_title('Task Arithmetic (TA) Sweeps')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')
    
    # 2. TIES and L-TIES
    ties_runs = [r for r in results if r["method"] == "TIES"]
    # Find best keep rate from TIES
    best_kr = 0.5 # Default/common best
    ties_subset = [r for r in ties_runs if float(r["params"].split(",")[0].split("=")[1]) == best_kr]
    ties_scales = sorted(list(set([float(r["params"].split(",")[1].split("=")[1]) for r in ties_subset])))
    
    uncal_ties = []
    cal_ties = []
    for scale in ties_scales:
        uncal_run = [r for r in ties_subset if not r["calibrated"] and float(r["params"].split(",")[1].split("=")[1]) == scale][0]
        cal_run = [r for r in ties_subset if r["calibrated"] and float(r["params"].split(",")[1].split("=")[1]) == scale][0]
        uncal_ties.append(uncal_run["mean_acc"])
        cal_ties.append(cal_run["mean_acc"])
        
    axes[1].plot(ties_scales, uncal_ties, 'o-', color='C0', label=f'Uncal TIES (kr={best_kr})')
    axes[1].plot(ties_scales, cal_ties, 's--', color='C1', label=f'Cal TIES (kr={best_kr})')
    
    # Add L-TIES points
    l_ties_runs = [r for r in results if r["method"] == "L-TIES"]
    for r in l_ties_runs:
        marker = '*' if 'Focused' in r["params"] else 'X'
        axes[1].plot(0.9, r["mean_acc"], marker, color='red', markersize=10,
                     label=f"L-TIES ({r['params'].split(' ')[0]})")
                     
    axes[1].set_xlabel('Scale Factor $\lambda$')
    axes[1].set_ylabel('Mean Accuracy (%)')
    axes[1].set_title(f'TIES-Merging Sweeps (keep_rate={best_kr})')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')
    
    # 3. DARE and L-DARE
    dare_runs = [r for r in results if r["method"] == "DARE"]
    best_dr = 0.1 # Default/common best
    dare_subset = [r for r in dare_runs if float(r["params"].split(",")[0].split("=")[1]) == best_dr]
    dare_scales = sorted(list(set([float(r["params"].split(",")[1].split("=")[1]) for r in dare_subset])))
    
    uncal_dare = []
    cal_dare = []
    for scale in dare_scales:
        uncal_run = [r for r in dare_subset if not r["calibrated"] and float(r["params"].split(",")[1].split("=")[1]) == scale][0]
        cal_run = [r for r in dare_subset if r["calibrated"] and float(r["params"].split(",")[1].split("=")[1]) == scale][0]
        uncal_dare.append(uncal_run["mean_acc"])
        cal_dare.append(cal_run["mean_acc"])
        
    axes[2].plot(dare_scales, uncal_dare, 'o-', color='C0', label=f'Uncal DARE (dr={best_dr})')
    axes[2].plot(dare_scales, cal_dare, 's--', color='C1', label=f'Cal DARE (dr={best_dr})')
    
    # Add L-DARE points
    l_dare_runs = [r for r in results if r["method"] == "L-DARE"]
    for r in l_dare_runs:
        marker = 'd' if 'Focused' in r["params"] else 'D'
        axes[2].plot(0.5, r["mean_acc"], marker, color='red', markersize=10,
                     label=f"L-DARE ({r['params'].split(' ')[0]})")
                     
    axes[2].set_xlabel('Scale Factor $\lambda$')
    axes[2].set_ylabel('Mean Accuracy (%)')
    axes[2].set_title(f'DARE-Merging Sweeps (drop_rate={best_dr})')
    axes[2].legend()
    axes[2].grid(True, linestyle=':')
    
    plt.tight_layout()
    plt.savefig("sweep_analysis.png", dpi=300)
    print("Saved sweep_analysis.png")

def plot_calibration_audit(audit_data):
    audit_results = audit_data["audit_results"]
    uncal_wa_acc = audit_data["uncal_wa_acc"]
    
    sample_sizes = sorted(list(audit_results.keys()))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Extract data for boxplots or error bars
    data_list = [audit_results[n] for n in sample_sizes]
    
    # Draw boxplots
    box = ax.boxplot(data_list, positions=range(len(sample_sizes)), patch_artist=True,
                     boxprops=dict(facecolor='C1', alpha=0.6),
                     medianprops=dict(color='black', linewidth=1.5),
                     whiskerprops=dict(color='C1', linewidth=1.5),
                     capprops=dict(color='C1', linewidth=1.5))
                     
    # Add individual points (jittered)
    for i, n in enumerate(sample_sizes):
        y = audit_results[n]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, color='darkorange', edgecolor='black', alpha=0.8, zorder=3)
        
    # Draw horizontal line for Uncalibrated WA
    ax.axhline(uncal_wa_acc, color='C0', linestyle='-', linewidth=2, label=f'Deterministic Uncalibrated WA ({uncal_wa_acc:.2f}%)')
    
    ax.set_xticklabels([str(n) for n in sample_sizes])
    ax.set_xlabel('Calibration Dataset Size $N$ (Samples per Task)')
    ax.set_ylabel('Mean Multi-Task Accuracy (%)')
    ax.set_title('Robustness Audit of Activation Calibration vs. Uncalibrated WA')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':')
    
    plt.tight_layout()
    plt.savefig("calibration_variance.png", dpi=300)
    print("Saved calibration_variance.png")

if __name__ == "__main__":
    try:
        data = torch.load("results_sweep_fast.pt", map_location='cpu')
        plot_sweeps(data)
    except Exception as e:
        print(f"Error plotting sweeps: {e}")
        
    try:
        audit_data = torch.load("calibration_audit_fast.pt", map_location='cpu')
        plot_calibration_audit(audit_data)
    except Exception as e:
        print(f"Error plotting calibration audit: {e}")
