import os
import json
import matplotlib.pyplot as plt

def main():
    summary_path = "results/summary.json"
    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}. Cannot generate plots yet.")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    # Output directory for plots
    os.makedirs("results", exist_ok=True)

    # Use standard professional style
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # Group data by bn_mode
    for bn in ["uniform", "expert"]:
        bn_data = [row for row in data if row["bn_mode"] == bn]
        if not bn_data:
            continue

        # Plot 1: Cosine Similarity vs. Weight Decay (epochs = 5)
        wd_data = sorted([row for row in bn_data if row["epochs"] == 5], key=lambda x: x["wd"])
        
        plt.figure(figsize=(5, 3.5))
        wds = [row["wd"] for row in wd_data]
        sims = [row["avg_similarity"] for row in wd_data]
        
        # Plot with log scale on X-axis (except 0.0, we will represent it as a tiny value or handle separately)
        # To avoid log(0), we can plot on a regular scale or just plot points
        x_labels = [str(w) for w in wds]
        plt.plot(range(len(wds)), sims, marker="o", color="#1f77b4", linewidth=2, markersize=8)
        plt.xticks(range(len(wds)), x_labels)
        plt.xlabel("Weight Decay")
        plt.ylabel("Avg Cosine Similarity")
        plt.title(f"Orthogonality vs. Regularization\n(Epochs = 5, BN = {bn.upper()})")
        plt.tight_layout()
        plt.savefig(f"results/cossim_vs_wd_bn_{bn}.pdf")
        plt.close()

        # Plot 2: Cosine Similarity vs. Fine-tuning Epochs (wd = 1e-2)
        # Note: the sweep had epochs = 1, 5, 20
        # Let's collect them from bn_data where wd = 1e-2
        epoch_data = sorted([row for row in bn_data if row["wd"] == 0.01], key=lambda x: x["epochs"])
        
        plt.figure(figsize=(5, 3.5))
        eps = [row["epochs"] for row in epoch_data]
        sims_ep = [row["avg_similarity"] for row in epoch_data]
        
        plt.plot(eps, sims_ep, marker="s", color="#d62728", linewidth=2, markersize=8)
        plt.xlabel("Fine-tuning Epochs")
        plt.ylabel("Avg Cosine Similarity")
        plt.title(f"Orthogonality vs. Fine-tuning Duration\n(WD = 0.01, BN = {bn.upper()})")
        plt.tight_layout()
        plt.savefig(f"results/cossim_vs_epochs_bn_{bn}.pdf")
        plt.close()

        # Plot 3: Merging Accuracy vs. Average Update Cosine Similarity
        # We will plot all sweep points on this plot to show the correlation
        sorted_by_sim = sorted(bn_data, key=lambda x: x["avg_similarity"])
        
        plt.figure(figsize=(6, 4))
        sim_vals = [row["avg_similarity"] for row in sorted_by_sim]
        
        plt.plot(sim_vals, [row["wa"] for row in sorted_by_sim], marker="o", label="Weight Averaging (WA)", linewidth=2)
        plt.plot(sim_vals, [row["ta_05"] for row in sorted_by_sim], marker="x", label="Task Arithmetic (TA, lambda=0.5)", linewidth=1.5, linestyle="--")
        plt.plot(sim_vals, [row["ta_07"] for row in sorted_by_sim], marker="v", label="Task Arithmetic (TA, lambda=0.7)", linewidth=1.5, linestyle="-.")
        plt.plot(sim_vals, [row["u_ipr"] for row in sorted_by_sim], marker="s", label="Update-level IPR (U-IPR)", linewidth=2)
        plt.plot(sim_vals, [row["hns"] for row in sorted_by_sim], marker="d", label="Holographic Norm Scaling (HNS)", linewidth=2)
        
        plt.xlabel("Avg Cosine Similarity (Increasing Collinearity)")
        plt.ylabel("Multi-Task Acc (%)")
        plt.title(f"Merging Method Robustness under Collinearity\n(BN Mode = {bn.upper()})")
        plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"results/acc_vs_cossim_bn_{bn}.pdf")
        plt.close()

    print("Successfully generated all analysis plots in PDF format.")

if __name__ == "__main__":
    main()
