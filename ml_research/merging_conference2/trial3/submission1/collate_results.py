import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def collate_results():
    # Find all result files
    files = glob.glob("results_sweep_*.json")
    if not files:
        print("No result files found yet.")
        return

    print(f"Found {len(files)} result files. Collating...")

    data = []
    for f in files:
        # Expected format: results_sweep_{merge_mode}_{coeff}_N{cal_size}_S{seed}.json
        # e.g., results_sweep_WA_0.3_N128_S42.json
        basename = os.path.basename(f).replace(".json", "")
        parts = basename.split("_")
        if len(parts) < 6:
            continue
        
        merge_mode = parts[2]
        coeff = float(parts[3])
        cal_size = int(parts[4][1:]) # Strip 'N'
        seed = int(parts[5][1:]) # Strip 'S'

        with open(f, "r") as fh:
            results = json.load(fh)
        
        for setup, task_accs in results.items():
            row = {
                "merge_mode": merge_mode,
                "coeff": coeff,
                "cal_size": cal_size,
                "seed": seed,
                "setup": setup,
                "mnist": task_accs["mnist"],
                "fmnist": task_accs["fmnist"],
                "cifar10": task_accs["cifar10"],
                "avg": task_accs["avg"]
            }
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("collated_raw_results.csv", index=False)
    print("Saved raw collated results to collated_raw_results.csv")

    # Aggregate across seeds
    group_cols = ["merge_mode", "coeff", "cal_size", "setup"]
    agg_df = df.groupby(group_cols).agg({
        "mnist": ["mean", "std"],
        "fmnist": ["mean", "std"],
        "cifar10": ["mean", "std"],
        "avg": ["mean", "std"]
    }).reset_index()

    # Flatten multi-index columns
    agg_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg_df.columns
    ]
    agg_df.to_csv("collated_aggregated_results.csv", index=False)
    print("Saved aggregated results to collated_aggregated_results.csv")

    # Generate Markdown summary of top-performing combinations
    print("\n=== TOP PERFORMING COMBINATIONS (AVG ACCURACY) ===")
    top_perf = agg_df.sort_values(by="avg_mean", ascending=False).head(15)
    print(top_perf[["merge_mode", "coeff", "cal_size", "setup", "avg_mean", "avg_std"]].to_markdown(index=False))

    # Generate Plots
    generate_plots(agg_df)

def generate_plots(df):
    os.makedirs("plots", exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'figure.titlesize': 14})

    # Setups to plot
    setups = [
        "baseline", "n_taac", "lsc", 
        "head_sft", "head_tta", 
        "n_taac_head_sft", "n_taac_head_tta", 
        "lsc_head_sft", "lsc_head_tta"
    ]
    
    # Beautiful styling
    colors = {
        "baseline": "black",
        "n_taac": "tab:blue",
        "lsc": "tab:orange",
        "head_sft": "tab:green",
        "head_tta": "tab:red",
        "n_taac_head_sft": "tab:purple",
        "n_taac_head_tta": "tab:brown",
        "lsc_head_sft": "tab:pink",
        "lsc_head_tta": "tab:cyan"
    }

    labels = {
        "baseline": "Baseline (No Calibration/Adaptation)",
        "n_taac": "N-TAAC Calibration",
        "lsc": "LSC Calibration",
        "head_sft": "Supervised Head SFT",
        "head_tta": "Unsupervised Head TTA",
        "n_taac_head_sft": "Synergy: N-TAAC + SFT",
        "n_taac_head_tta": "Synergy: N-TAAC + TTA",
        "lsc_head_sft": "Synergy: LSC + SFT",
        "lsc_head_tta": "Synergy: LSC + TTA"
    }

    # Plot 1: WA - Calibration Size N vs. Avg Accuracy
    df_wa = df[df["merge_mode"] == "WA"]
    if not df_wa.empty:
        plt.figure(figsize=(10, 6))
        for setup in setups:
            sub = df_wa[df_wa["setup"] == setup].sort_values(by="cal_size")
            if sub.empty:
                continue
            plt.errorbar(
                sub["cal_size"], sub["avg_mean"], yerr=sub["avg_std"],
                label=labels[setup], color=colors[setup], marker="o", capsize=4, linewidth=2
            )
        plt.xscale("log")
        plt.xticks([4, 8, 16, 32, 64, 128, 256], [4, 8, 16, 32, 64, 128, 256])
        plt.xlabel("Calibration Sample Size (N) per Task")
        plt.ylabel("Multi-Task Average Accuracy (%)")
        plt.title("WA Merging: Calibration Budget vs. Multi-Task Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("plots/wa_calibration_budget_vs_accuracy.png", dpi=300)
        plt.close()

    # Plot 2: TA - Merging Coefficient lambda vs. Avg Accuracy (for N=128)
    df_ta_n128 = df[(df["merge_mode"] == "TA") & (df["cal_size"] == 128)]
    if not df_ta_n128.empty:
        plt.figure(figsize=(10, 6))
        for setup in setups:
            sub = df_ta_n128[df_ta_n128["setup"] == setup].sort_values(by="coeff")
            if sub.empty:
                continue
            plt.errorbar(
                sub["coeff"], sub["avg_mean"], yerr=sub["avg_std"],
                label=labels[setup], color=colors[setup], marker="s", capsize=4, linewidth=2
            )
        plt.xlabel("Task Arithmetic Scaling Coefficient ($\lambda$)")
        plt.ylabel("Multi-Task Average Accuracy (%)")
        plt.title("TA Merging (N=128): Scaling Coefficient vs. Multi-Task Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("plots/ta_scaling_vs_accuracy.png", dpi=300)
        plt.close()

    # Plot 3: TA - Calibration Size N vs. Avg Accuracy (for best coeff say 0.3)
    df_ta_c03 = df[(df["merge_mode"] == "TA") & (df["coeff"] == 0.3)]
    if not df_ta_c03.empty:
        plt.figure(figsize=(10, 6))
        for setup in setups:
            sub = df_ta_c03[df_ta_c03["setup"] == setup].sort_values(by="cal_size")
            if sub.empty:
                continue
            plt.errorbar(
                sub["cal_size"], sub["avg_mean"], yerr=sub["avg_std"],
                label=labels[setup], color=colors[setup], marker="^", capsize=4, linewidth=2
            )
        plt.xscale("log")
        plt.xticks([4, 8, 16, 32, 64, 128, 256], [4, 8, 16, 32, 64, 128, 256])
        plt.xlabel("Calibration Sample Size (N) per Task")
        plt.ylabel("Multi-Task Average Accuracy (%)")
        plt.title("TA Merging ($\lambda=0.3$): Calibration Budget vs. Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("plots/ta_calibration_budget_vs_accuracy.png", dpi=300)
        plt.close()

    print("Generated all publication plots in `./plots` directory.")

if __name__ == "__main__":
    collate_results()
