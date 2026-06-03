import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_all_results():
    results = []
    # Find all json files in results
    for path in glob.glob("results/*.json"):
        filename = os.path.basename(path).replace(".json", "")
        # Parse configuration from filename
        # Format: {merge_method}_lam{ta_lam}_rep{rep_cal}_head{head_align}_N{cal_size}_seed{seed}
        parts = filename.split("_")
        if len(parts) < 6:
            continue
        merge_method = parts[0]
        ta_lam = float(parts[1].replace("lam", ""))
        rep_cal = parts[2].replace("rep", "")
        head_align = parts[3].replace("head", "")
        cal_size = int(parts[4].replace("N", ""))
        seed = int(parts[5].replace("seed", ""))
        
        with open(path, "r") as f:
            data = json.load(f)
            
        results.append({
            'merge_method': merge_method,
            'ta_lam': ta_lam,
            'rep_cal': rep_cal,
            'head_align': head_align,
            'cal_size': cal_size,
            'seed': seed,
            'mnist': data['mnist'],
            'fmnist': data['fmnist'],
            'cifar10': data['cifar10'],
            'avg': data['avg']
        })
    return results

def print_summary_table(results):
    # Main matrix has N=128
    main_results = [r for r in results if r['cal_size'] == 128]
    
    # We want to aggregate across seeds
    configs = {}
    for r in main_results:
        key = (r['merge_method'], r['rep_cal'], r['head_align'])
        if key not in configs:
            configs[key] = []
        configs[key].append(r['avg'])
        
    print("\n" + "="*50)
    print("MAIN MATRIX ACCURACY SUMMARY (N=128) ACROSS SEEDS")
    print("="*50)
    print(f"{'Merge':<6} | {'Rep Cal':<8} | {'Head Align':<10} | {'Mean Acc (%)':<15} | {'Std Dev':<8}")
    print("-"*50)
    
    # Format for markdown output as well
    md_lines = [
        "| Merge Method | Representation Cal | Head Alignment | Mean Accuracy (%) | Std Dev |",
        "|---|---|---|---|---|"
    ]
    
    for key, accs in sorted(configs.items()):
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{key[0].upper():<6} | {key[1].upper():<8} | {key[2].upper():<10} | {mean_acc:.2f}% | {std_acc:.2f}")
        md_lines.append(f"| {key[0].upper()} | {key[1].upper()} | {key[2].upper()} | {mean_acc:.2f}% | {std_acc:.2f} |")
        
    # Write to a text file for easy reading
    with open("results_summary.md", "w") as f:
        f.write("# Experimental Results Summary\n\n")
        f.write("\n".join(md_lines))
        f.write("\n")

def plot_sample_efficiency(results):
    # We plot WA across N in {16, 32, 64, 128, 256, 512}
    configs_to_plot = [
        {"rep_cal": "none", "head_align": "none", "label": "Baseline (Uncalibrated)", "color": "gray", "marker": "o", "ls": "--"},
        {"rep_cal": "none", "head_align": "sft", "label": "Head SFT Alone", "color": "blue", "marker": "^", "ls": "-"},
        {"rep_cal": "none", "head_align": "tta", "label": "Head TTA Alone", "color": "cyan", "marker": "v", "ls": "-"},
        {"rep_cal": "ntaac", "head_align": "none", "label": "N-TAAC Alone", "color": "green", "marker": "s", "ls": "-"},
        {"rep_cal": "ntaac", "head_align": "sft", "label": "N-TAAC + Head SFT (Ours)", "color": "red", "marker": "D", "ls": "-"},
        {"rep_cal": "ntaac", "head_align": "tta", "label": "N-TAAC + Head TTA (Ours)", "color": "orange", "marker": "*", "ls": "-"}
    ]
    
    plt.figure(figsize=(8, 6))
    
    for conf in configs_to_plot:
        # Find results for this config with merge_method="wa"
        subset = [
            r for r in results 
            if r['merge_method'] == "wa" 
            and r['rep_cal'] == conf['rep_cal'] 
            and r['head_align'] == conf['head_align']
        ]
        
        if not subset:
            continue
            
        # Group by cal_size
        sizes = sorted(list(set([r['cal_size'] for r in subset])))
        x = []
        y_mean = []
        y_std = []
        
        for size in sizes:
            size_subset = [r for r in subset if r['cal_size'] == size]
            accs = [r['avg'] for r in size_subset]
            x.append(size)
            y_mean.append(np.mean(accs))
            y_std.append(np.std(accs) if len(accs) > 1 else 0.0)
            
        x = np.array(x)
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)
        
        # If it's the baseline (repnone_headnone), it doesn't use calibration, so N doesn't matter.
        if conf['rep_cal'] == "none" and conf['head_align'] == "none":
            baseline_val = y_mean[0]
            plt.axhline(y=baseline_val, color=conf['color'], linestyle=conf['ls'], label=conf['label'])
        else:
            plt.plot(x, y_mean, label=conf['label'], color=conf['color'], marker=conf['marker'], linestyle=conf['ls'], linewidth=2, markersize=8)
            if np.any(y_std > 0):
                plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=conf['color'], alpha=0.15)
            
    plt.xscale('log')
    plt.xticks([16, 32, 64, 128, 256, 512], ['16', '32', '64', '128', '256', '512'])
    plt.xlabel('Calibration Sample Size ($N$ per task)', fontsize=12)
    plt.ylabel('Average Multi-Task Accuracy (%)', fontsize=12)
    plt.title('Sample Efficiency: Representation vs. Decision Boundary Alignment', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls=":")
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    
    # Save plot
    plt.savefig("sample_efficiency.png", dpi=300)
    print("Saved sample efficiency plot to sample_efficiency.png")
    
    # Also generate a table of these sample efficiency values for the LaTeX document
    print("\n" + "="*50)
    print("SAMPLE EFFICIENCY DETAILED VALUES (WA) ACROSS SEEDS (Mean +- Std)")
    print("="*50)
    for conf in configs_to_plot:
        subset = [
            r for r in results 
            if r['merge_method'] == "wa" 
            and r['rep_cal'] == conf['rep_cal'] 
            and r['head_align'] == conf['head_align']
        ]
        if not subset:
            continue
        sizes = sorted(list(set([r['cal_size'] for r in subset])))
        vals = []
        for size in sizes:
            size_subset = [r for r in subset if r['cal_size'] == size]
            accs = [r['avg'] for r in size_subset]
            mean_val = np.mean(accs)
            std_val = np.std(accs) if len(accs) > 1 else 0.0
            vals.append(f"N={size}:{mean_val:.2f}% (+-{std_val:.2f})")
        print(f"{conf['label']:<25} | {', '.join(vals)}")

def main():
    results = load_all_results()
    if not results:
        print("No results found yet.")
        return
        
    print(f"Loaded {len(results)} experimental results.")
    print_summary_table(results)
    plot_sample_efficiency(results)

if __name__ == "__main__":
    main()
