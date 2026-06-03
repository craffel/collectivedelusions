import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    json_files = glob.glob("results/lambda_sweep/*.json")
    print(f"Found {len(json_files)} result files.")
    
    # Store structure: {config_name: {lambda: [seed1_val, seed2_val, seed3_val]}}
    data = {
        "uncalibrated": {},
        "ntaac_only": {},
        "tta_only": {},
        "reda": {}
    }
    
    configs = {
        "repnone_headnone": "uncalibrated",
        "repntaac_headnone": "ntaac_only",
        "repnone_headtta": "tta_only",
        "repntaac_headtta": "reda"
    }
    
    for file_path in json_files:
        basename = os.path.basename(file_path).replace(".json", "")
        # Format: ta_lam{lambda}_rep{rep}_head{head}_N128_seed{seed}
        parts = basename.split("_")
        
        # Extract lambda
        # Part 1 is lam{val}
        try:
            ta_lam = float(parts[1].replace("lam", ""))
        except ValueError:
            continue
            
        # Match config key
        config_key = f"{parts[2]}_{parts[3]}"
        if config_key not in configs:
            continue
        config_name = configs[config_key]
        
        with open(file_path, "r") as f:
            res = json.load(f)
            avg_acc = res["avg"]
            
        if ta_lam not in data[config_name]:
            data[config_name][ta_lam] = []
        data[config_name][ta_lam].append(avg_acc)
        
    # Aggregate and sort
    results = {}
    for config_name, lam_dict in data.items():
        sorted_lams = sorted(lam_dict.keys())
        means = []
        stds = []
        for lam in sorted_lams:
            vals = lam_dict[lam]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        results[config_name] = {
            "lambdas": sorted_lams,
            "means": means,
            "stds": stds
        }
        
    # Plot results
    plt.figure(figsize=(8, 5.5))
    
    colors = {
        "uncalibrated": "#7f7f7f",
        "ntaac_only": "#1f77b4",
        "tta_only": "#ff7f0e",
        "reda": "#d62728"
    }
    
    labels = {
        "uncalibrated": "Uncalibrated TA",
        "ntaac_only": "N-TAAC alone",
        "tta_only": "Head TTA alone",
        "reda": "REDA (N-TAAC + Head TTA) [Ours]"
    }
    
    for config_name, res in results.items():
        lams = np.array(res["lambdas"])
        means = np.array(res["means"])
        stds = np.array(res["stds"])
        
        plt.plot(lams, means, label=labels[config_name], color=colors[config_name], linewidth=2.5)
        plt.fill_between(lams, means - stds, means + stds, color=colors[config_name], alpha=0.15)
        
    plt.xlabel("Task Arithmetic Scaling Coefficient ($\lambda$)", fontsize=12)
    plt.ylabel("Multi-Task Mean Accuracy (%)", fontsize=12)
    plt.title("Sensitivity to Task Arithmetic Scaling Coefficient ($\lambda$)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10.5, loc="lower left")
    plt.xlim(0.0, 1.5)
    plt.ylim(0.0, 80.0)
    plt.tight_layout()
    
    plt.savefig("lambda_sensitivity.png", dpi=300)
    print("Saved plot to lambda_sensitivity.png")
    
    # Print Markdown table
    print("\n| Scaling Coefficient ($\lambda$) | Uncalibrated (%) | N-TAAC alone (%) | Head TTA alone (%) | REDA [Ours] (%) |")
    print("|---|---|---|---|---|")
    for idx, lam in enumerate(results["reda"]["lambdas"]):
        row_str = f"| {lam:.1f} "
        for config_name in ["uncalibrated", "ntaac_only", "tta_only", "reda"]:
            mean_val = results[config_name]["means"][idx]
            std_val = results[config_name]["stds"][idx]
            row_str += f"| {mean_val:.2f} ({std_val:.2f}) "
        row_str += "|"
        print(row_str)

if __name__ == "__main__":
    main()
