"""Plot the routing diagnostic (cycle-4)."""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    in_path = "/fsx/craffel/collectivedelusions/ml_research/testclaude/results/routing_diagnostic.json"
    out_path = "/fsx/craffel/collectivedelusions/ml_research/testclaude/results/routing_diagnostic.png"

    with open(in_path) as f:
        data = json.load(f)
    null_base = data.get("null_baseline_expected", 0.1)
    ks_list = data["keep_fracs"]
    ks = sorted(ks_list, reverse=True)

    def get(d, keys):
        return [d["per_k"][f"k{k}"][keys] for k in ks]

    no_sign = data["no_sign"]
    on_ns  = get(no_sign, "mean_on_mass_frac")
    off_ns = get(no_sign, "mean_off_mass_frac")
    ratio_ns = get(no_sign, "on_over_off")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    ax.plot(ks, on_ns, marker='o', color='C0', linewidth=2,
            label=r"On-target cols (task $t$'s active cols)")
    ax.plot(ks, off_ns, marker='s', color='C1', linewidth=2,
            label=r"Off-target cols (other tasks' active cols)")
    ax.axhline(y=null_base, color='gray', linestyle=':', alpha=0.7, label='Null baseline (uniform routing)')
    ax.set_xlabel(r'Keep-fraction $k$')
    ax.set_ylabel(r'Mass of $P_t = C_t \Sigma^{-1}$ on columns (frac.)')
    ax.set_title(r'(a) Routing-mass distribution')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

    ax = axes[1]
    ax.plot(ks, ratio_ns, marker='D', color='C3', linewidth=2, label='on/off ratio')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Null baseline (=1)')
    for x, r in zip(ks, ratio_ns):
        if x in (1.0, 0.3, 0.05):
            ax.annotate(f'{r:.2f}', xy=(x, r), xytext=(4, 4), textcoords='offset points', fontsize=9)
    ax.set_xlabel(r'Keep-fraction $k$')
    ax.set_ylabel('Ratio: on-target mass / off-target mass')
    ax.set_title(r'(b) Task-routing concentration')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
