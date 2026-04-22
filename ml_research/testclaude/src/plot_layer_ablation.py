"""Plot the per-layer ablation results."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN = "/fsx/craffel/collectivedelusions/ml_research/testclaude/results/layer_ablation.json"
OUT = "/fsx/craffel/collectivedelusions/ml_research/testclaude/results/layer_ablation.png"
ACTMAT_AVG = 0.780  # from main table
SCALE_AVG = 0.845   # from main table


def main():
    with open(IN) as f:
        r = json.load(f)

    # Two panels: block type, block range
    type_order = [
        ("all_layers", "SCALE everywhere"),
        ("mlp_in_only", "MLP-in only"),
        ("mlp_out_only", "MLP-out only"),
        ("attn_qkv_only", "Attn QKV only"),
        ("attn_out_only", "Attn-out only"),
        ("mlp_only", "All MLP only"),
        ("attn_only", "All Attn only"),
    ]
    range_order = [
        ("all_layers", "All blocks (0-11)"),
        ("early_blocks_0_4", "Early (0-3)"),
        ("middle_blocks_4_8", "Middle (4-7)"),
        ("late_blocks_8_12", "Late (8-11)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    for ax, (title, config) in zip(axes, [("Layer type", type_order), ("Block range", range_order)]):
        labels = []
        avgs = []
        for key, label in config:
            if key in r:
                labels.append(label)
                avgs.append(r[key]["avg"] * 100)
        y = np.arange(len(labels))
        colors = ["#1f77b4" if labels[i] in ("SCALE everywhere", "All blocks (0-11)") else "#87ceeb" for i in range(len(labels))]
        ax.barh(y, avgs, color=colors, edgecolor="black")
        ax.axvline(ACTMAT_AVG * 100, ls="--", color="red", alpha=0.6, label=f"ACTMat ({ACTMAT_AVG*100:.1f}%)")
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlabel("Average merged accuracy (%)")
        ax.set_title(title)
        ax.set_xlim(75, 87)
        ax.invert_yaxis()
        for i, v in enumerate(avgs):
            ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=9)
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
