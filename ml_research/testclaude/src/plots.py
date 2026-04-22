"""Generate plots from results.json."""
import argparse, json, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bar_plot(results_path, out_path):
    with open(results_path) as f:
        results = json.load(f)
    tasks = results["tasks"]
    methods = results["methods"]
    ordered = ["pretrained", "individual", "simple_avg", "task_arith_best", "della_best",
               "breadcrumbs_best", "ties_best", "consensus_best", "actmat", "scale_best"]
    pretty = {"pretrained": "Pretrained (zero-shot)", "individual": "Individual (upper bound)",
              "simple_avg": "Simple Avg", "task_arith_best": "Task Arithmetic",
              "della_best": "DELLA",
              "breadcrumbs_best": "Breadcrumbs",
              "ties_best": "TIES-Merging", "consensus_best": "Consensus",
              "actmat": "ACTMat (data-free)",
              "scale_best": "SCALE (ours)"}
    avail = [m for m in ordered if m in methods]
    fig, ax = plt.subplots(figsize=(12, 5))
    avgs = [methods[m]["avg"] for m in avail]
    colors = ["#888", "#444", "#1f77b4", "#2ca02c", "#17becf", "#8c564b",
              "#9467bd", "#e377c2", "#ff7f0e", "#d62728"]
    bars = ax.bar([pretty[m] for m in avail], avgs, color=colors[:len(avail)])
    ax.set_ylabel("Average accuracy across tasks")
    ax.set_ylim(0, 1)
    for b, a in zip(bars, avgs):
        ax.text(b.get_x() + b.get_width()/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def per_task_plot(results_path, out_path):
    with open(results_path) as f:
        results = json.load(f)
    tasks = results["tasks"]
    methods = results["methods"]
    ordered = ["task_arith_best", "della_best", "breadcrumbs_best", "ties_best",
               "consensus_best", "actmat", "scale_best"]
    pretty = {"simple_avg": "Simple Avg", "task_arith_best": "Task Arithmetic",
              "della_best": "DELLA",
              "breadcrumbs_best": "Breadcrumbs",
              "ties_best": "TIES", "consensus_best": "Consensus",
              "actmat": "ACTMat",
              "scale_best": "SCALE (ours)"}
    avail = [m for m in ordered if m in methods]
    width = 0.115
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, m in enumerate(avail):
        vals = [methods[m]["per_task"][t] for t in tasks]
        ax.bar(x + i * width, vals, width, label=pretty[m])
    ax.set_xticks(x + (len(avail) - 1) * width / 2)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def keep_sweep_plot(abl_path, out_path):
    if not os.path.exists(abl_path):
        return
    with open(abl_path) as f:
        a = json.load(f)
    m = a["methods"]
    ks = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    sign_avg, nosign_avg = [], []
    for k in ks:
        s = m.get(f"scale_sign_k{k}")
        n = m.get(f"scale_nosign_k{k}")
        sign_avg.append(s["avg"] if s else np.nan)
        nosign_avg.append(n["avg"] if n else np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, sign_avg, "-o", color="#d62728", label="SCALE (with sign election)")
    ax.plot(ks, nosign_avg, "-s", color="#ff7f0e", label="SCALE (no sign election)")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Keep fraction $k$ (top-$k$ magnitude)")
    ax.set_ylabel("Average accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--ablations", default="results/ablations.json")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.exists(args.results):
        bar_plot(args.results, os.path.join(args.out_dir, "avg_accuracy.png"))
        per_task_plot(args.results, os.path.join(args.out_dir, "per_task.png"))
    keep_sweep_plot(args.ablations, os.path.join(args.out_dir, "keep_sweep.png"))
    print("Plots saved.")


if __name__ == "__main__":
    main()
