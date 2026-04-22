"""Plot angular distance & subspace alignment as a function of trim k."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/fsx/craffel/collectivedelusions/ml_research/testclaude/results"


def main():
    agg = json.load(open(os.path.join(RESULTS, "angular_agg.json")))
    per = json.load(open(os.path.join(RESULTS, "angular_validation.json")))
    ks = agg["k_values"]

    # Collect per-k angle (sign=0) and subspace overlap (sign=0)
    # Summed
    summed_angle = [agg["summed_angle_mean"][f"k={k},sign=0"] for k in ks]
    subspace_overlap = [agg["subspace_alignment_mean"][f"k={k},sign=0"] for k in ks]
    # Per-task (sign=0) — average across all tasks/layers from per["summary"]
    per_task_angle = [per["summary"]["averages"][f"k={k},sign=0"]["mean_deg"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    ax = axes[0]
    ax.plot(ks, per_task_angle, "o-", label=r"per-task $\angle(\Delta_t^\top\Delta_t, C_t)$", color="tab:blue")
    ax.plot(ks, summed_angle, "s-", label=r"summed $\angle(\Sigma_t\tilde\Delta_t^\top\tilde\Delta_t, \Sigma_t C_t)$", color="tab:orange")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"keep fraction $k$")
    ax.set_ylabel(r"angular distance (deg)")
    ax.set_title("Data-free covariance approximation\ndoes not improve with trimming")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.plot(ks, subspace_overlap, "^-", color="tab:green")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"keep fraction $k$")
    ax.set_ylabel(r"top-32 subspace alignment")
    ax.set_title(r"Top-32 eigenspace overlap between $\Sigma\tilde\Delta_t^\top\tilde\Delta_t$ and $\Sigma C_t$")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "angular_distance.png"), dpi=160, bbox_inches="tight")
    print("Saved", os.path.join(RESULTS, "angular_distance.png"))


if __name__ == "__main__":
    main()
