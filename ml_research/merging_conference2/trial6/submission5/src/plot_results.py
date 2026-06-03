import os
import json
import matplotlib.pyplot as plt

os.makedirs("models", exist_ok=True)

def generate_plots():
    results_path = "models/evaluation_results.json"
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found. Skip plotting.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    expert_baselines = data["expert_baselines"]
    avg_expert = sum(expert_baselines.values()) / 3 * 100
    
    results_data = data["results_data"]
    robustness_results = data["robustness_results"]

    # --- Plot 1: SP-TTBC Performance vs Alpha for WA and TA ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    for i, name in enumerate(["WA", "TA"]):
        ax = axes[i]
        method_data = results_data[name]
        all_sweep = method_data["all_sweep_results"]
        wa_no_cal = method_data["no_calibration"]
        wa_offline = method_data["offline_calibration"]
        
        # Group results by batch size
        by_bs = {}
        for res in all_sweep:
            bs = res["batch_size"]
            if bs not in by_bs:
                by_bs[bs] = []
            by_bs[bs].append((res["alpha"], res["avg"]))

        for bs in sorted(by_bs.keys()):
            by_bs[bs].sort()
            alphas = [x[0] for x in by_bs[bs]]
            accs = [x[1] * 100 for x in by_bs[bs]]
            ax.plot(alphas, accs, marker="o", linewidth=1.5, markersize=5, label=f"Test Batch Size: {bs}")

        avg_no_cal = sum(wa_no_cal.values()) / 3 * 100
        avg_offline = sum(wa_offline.values()) / 3 * 100
        
        ax.axhline(y=avg_no_cal, color="r", linestyle="--", linewidth=1.2, label="No Calibration")
        ax.axhline(y=avg_offline, color="g", linestyle="-.", linewidth=1.2, label="Offline Cal (REPAIR)")
        ax.axhline(y=avg_expert, color="black", linestyle=":", linewidth=1.2, label="Expert Oracle")
        
        ax.set_title(f"({chr(65+i)}) {name} Backbone: SP-TTBC Accuracy", fontsize=12, fontweight="bold")
        ax.set_xlabel("Blending Factor (Alpha)", fontsize=11)
        ax.set_ylabel("Average Multi-Task Accuracy (%)", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig("models/sp_ttbc_curves.png", dpi=300)
    plt.savefig("template/sp_ttbc_curves.png", dpi=300)
    plt.close()
    print("Generated sp_ttbc_curves.png")

    # --- Plot 2: Grouped Bar Chart Comparing WA and TA ---
    categories = ["No Calibration", "Offline Cal (REPAIR)", "SP-TTBC (Ours)"]
    
    # Extract WA results
    wa_nc = sum(results_data["WA"]["no_calibration"].values()) / 3 * 100
    wa_oc = sum(results_data["WA"]["offline_calibration"].values()) / 3 * 100
    wa_tt = results_data["WA"]["best_sp_ttbc"]["avg"] * 100
    
    # Extract TA results
    ta_nc = sum(results_data["TA"]["no_calibration"].values()) / 3 * 100
    ta_oc = sum(results_data["TA"]["offline_calibration"].values()) / 3 * 100
    ta_tt = results_data["TA"]["best_sp_ttbc"]["avg"] * 100

    wa_means = [wa_nc, wa_oc, wa_tt]
    ta_means = [ta_nc, ta_oc, ta_tt]

    import numpy as np
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rects1 = ax.bar(x - width/2, wa_means, width, label="Weight Averaging (WA)", color="#DD8452")
    rects2 = ax.bar(x + width/2, ta_means, width, label="Task Arithmetic (TA)", color="#4C72B0")
    
    ax.axhline(y=avg_expert, color="black", linestyle=":", linewidth=1.5, label="Expert Oracle (93.20%)")

    ax.set_ylabel("Average Multi-Task Accuracy (%)", fontsize=11)
    ax.set_title("Multi-Task Merging Calibration Performance", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig("models/calibration_comparison.png", dpi=300)
    plt.savefig("template/calibration_comparison.png", dpi=300)
    plt.close()
    print("Generated calibration_comparison.png")

    # --- Plot 3: Robustness under Covariate Shift (Gaussian Noise & Brightness Shift) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Left Column: Gaussian Noise (Top WA, Bottom TA)
    # Right Column: Brightness Shift (Top WA, Bottom TA)
    
    for row_idx, name in enumerate(["WA", "TA"]):
        # 1. Gaussian Noise (Row row_idx, Col 0)
        ax_gn = axes[row_idx, 0]
        gn_data = robustness_results["gaussian_noise"]
        levels_gn = gn_data["levels"]
        
        ax_gn.plot(levels_gn, [x * 100 for x in gn_data[name]["No_Cal"]], 'ro-', linewidth=2, label="No Calibration")
        ax_gn.plot(levels_gn, [x * 100 for x in gn_data[name]["Offline_Cal"]], 'g^-', linewidth=2, label="Offline Cal (REPAIR)")
        ax_gn.plot(levels_gn, [x * 100 for x in gn_data[name]["SP_TTBC"]], 'b*-', linewidth=2, label="SP-TTBC (Ours)")
        
        ax_gn.set_title(f"({chr(65 + row_idx*2)}) {name} Backbone: Robustness to Gaussian Noise", fontsize=11, fontweight="bold")
        ax_gn.set_xlabel("Noise Std Deviation (Sigma)", fontsize=10)
        ax_gn.set_ylabel("Average Accuracy (%)", fontsize=10)
        ax_gn.grid(True, linestyle=":", alpha=0.5)
        ax_gn.legend(fontsize=9, loc="lower left")
        ax_gn.set_ylim(-5, 105)

        # 2. Brightness Shift (Row row_idx, Col 1)
        ax_bs = axes[row_idx, 1]
        bs_data = robustness_results["brightness_shift"]
        levels_bs = bs_data["levels"]
        
        ax_bs.plot(levels_bs, [x * 100 for x in bs_data[name]["No_Cal"]], 'ro-', linewidth=2, label="No Calibration")
        ax_bs.plot(levels_bs, [x * 100 for x in bs_data[name]["Offline_Cal"]], 'g^-', linewidth=2, label="Offline Cal (REPAIR)")
        ax_bs.plot(levels_bs, [x * 100 for x in bs_data[name]["SP_TTBC"]], 'b*-', linewidth=2, label="SP-TTBC (Ours)")
        
        ax_bs.set_title(f"({chr(66 + row_idx*2)}) {name} Backbone: Robustness to Brightness Shift", fontsize=11, fontweight="bold")
        ax_bs.set_xlabel("Brightness Shift Intensity (Delta)", fontsize=10)
        ax_bs.set_ylabel("Average Accuracy (%)", fontsize=10)
        ax_bs.grid(True, linestyle=":", alpha=0.5)
        ax_bs.legend(fontsize=9, loc="lower left")
        ax_bs.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig("models/robustness_comparison.png", dpi=300)
    plt.savefig("template/robustness_comparison.png", dpi=300)
    plt.close()
    print("Generated robustness_comparison.png")

if __name__ == "__main__":
    generate_plots()
