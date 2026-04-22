"""Plots for Phase-5 refinement experiments (extras.json)."""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ridge_plot(path, out):
    with open(path) as f:
        d = json.load(f)
    s = d.get("ridge_sweep", {})
    if not s:
        return
    items = sorted(s.values(), key=lambda x: x["rho"])
    rhos = [it["rho"] for it in items]
    avgs = [it["avg"] for it in items]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(rhos, avgs, "-o", color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel(r"Ridge $\rho$")
    ax.set_ylabel("Average accuracy")
    ax.grid(alpha=0.3)
    for rh, av in zip(rhos, avgs):
        ax.annotate(f"{av:.3f}", (rh, av), fontsize=8, xytext=(0, 6), textcoords="offset points", ha="center")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def taskcount_plot(path, out):
    with open(path) as f:
        d = json.load(f)
    tc = d.get("task_count", {})
    if not tc:
        return
    Ts = [2, 4, 6, 8]
    methods = [
        ("task_arith", "Task Arithmetic", "#2ca02c"),
        ("ties", "TIES", "#9467bd"),
        ("actmat", "ACTMat", "#ff7f0e"),
        ("scale", "SCALE (ours)", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, lab, color in methods:
        avgs = [tc.get(f"{key}_T{T}", {}).get("avg", np.nan) for T in Ts]
        ax.plot(Ts, avgs, "-o", color=color, label=lab, linewidth=2)
    ax.set_xlabel("Number of merged tasks $T$")
    ax.set_ylabel("Average accuracy on merged tasks")
    ax.set_xticks(Ts)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def sign_agree_plot(path, out):
    with open(path) as f:
        d = json.load(f)
    sa = d.get("sign_agreement", {})
    if not sa:
        return
    ks, vs = [], []
    for k, v in sa.items():
        try:
            ks.append(float(k[1:])); vs.append(v)
        except Exception:
            continue
    order = np.argsort(ks)
    ks = np.array(ks)[order]; vs = np.array(vs)[order]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, vs, "-o", color="#8c564b")
    ax.axhline(0.5, linestyle=":", color="grey", label="chance")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Keep fraction $k$ (top-$k$ magnitude trim)")
    ax.set_ylabel("Majority-sign agreement rate")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    for kk, vv in zip(ks, vs):
        ax.annotate(f"{vv:.2f}", (kk, vv), fontsize=8, xytext=(0, 6), textcoords="offset points", ha="center")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extras", default="results/extras.json")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    if not os.path.exists(args.extras):
        print("No extras.json yet.")
        return
    ridge_plot(args.extras, os.path.join(args.out_dir, "ridge_sweep.png"))
    taskcount_plot(args.extras, os.path.join(args.out_dir, "taskcount.png"))
    sign_agree_plot(args.extras, os.path.join(args.out_dir, "sign_agreement.png"))
    print("Extra plots saved.")


if __name__ == "__main__":
    main()
