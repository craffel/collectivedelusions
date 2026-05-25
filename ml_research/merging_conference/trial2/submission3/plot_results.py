import json
import matplotlib.pyplot as plt
import numpy as np

# Load the sweep and diagnostic results
with open("./results/sweep_diagnostics.json", "r") as f:
    data = json.load(f)

sweep = data["sweep"]
diagnostics = data["diagnostics"]

# Set style for academic paper - highly compact and clean
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
    "text.usetex": False
})

# ----------------------------------------------------------------------
# Figure 1: Diagnostic Traces (MNIST classification head across 10 steps)
# ----------------------------------------------------------------------
steps = list(range(11))

# Shrink figsize to be compact
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 2.7))

methods = ["SyMerge", "ISR-TTA", "SOSR-TTA"]
labels = ["Standard SyMerge", r"ISR-TTA ($\beta=0.1$)", r"SOSR-TTA ($\beta=0.1$)"]
colors = ["#d62728", "#1f77b4", "#2ca02c"]
markers = ["x", "o", "s"]

# Left Subplot: Mean absolute cosine similarity (Class overlap)
for method, label, color, marker in zip(methods, labels, colors, markers):
    trace = diagnostics[method]
    cos_sims = [step_log["MNIST"]["mean_abs_cos_sim"] for step_log in trace]
    ax1.plot(steps, cos_sims, label=label, color=color, marker=marker, markersize=4)

ax1.set_xlabel("TTA Optimization Step")
ax1.set_ylabel("Mean Prototype Cosine Similarity")
ax1.set_title("Class Prototype Overlap (MNIST Head)")
ax1.set_xticks(steps)
ax1.grid(True)
ax1.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.8)

# Right Subplot: Condition Number (Spectral collapse)
for method, label, color, marker in zip(methods, labels, colors, markers):
    trace = diagnostics[method]
    cond_nums = [step_log["MNIST"]["condition_number"] for step_log in trace]
    ax2.plot(steps, cond_nums, label=label, color=color, marker=marker, markersize=4)

ax2.set_xlabel("TTA Optimization Step")
ax2.set_ylabel("Singular Value Condition Number")
ax2.set_title("Spectral Decay and Collapse")
ax2.set_xticks(steps)
ax2.grid(True)
ax2.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.8)

plt.tight_layout()
plt.savefig("./diagnostics_trace.pdf", dpi=300)
plt.close()
print("Saved diagnostics_trace.pdf (compact version)")

# ----------------------------------------------------------------------
# Figure 2: Beta Sensitivity Sweep
# ----------------------------------------------------------------------
betas = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
corruptions = ["clean", "noise", "rotation"]
titles = ["Clean Adaptation", "OOD Gaussian Noise", "OOD Image Rotation"]

fig2, axes = plt.subplots(1, 3, figsize=(11.5, 2.7))

for idx, (corr, title) in enumerate(zip(corruptions, titles)):
    ax = axes[idx]
    
    # Extract values for ISR-TTA
    isr_vals = []
    for b in betas:
        isr_vals.append(sweep[corr]["isr-fo"][str(b)]["Average"])
        
    # Extract values for SOSR-TTA
    sosr_vals = []
    for b in betas:
        sosr_vals.append(sweep[corr]["sosr"][str(b)]["Average"])
        
    symerge_baseline = {"clean": 55.89, "noise": 30.70, "rotation": 27.56}[corr]
    
    ax.axhline(y=symerge_baseline, color="gray", linestyle="--", label="Standard SyMerge")
    ax.plot(betas, isr_vals, label="ISR-TTA (Ours)", color="#1f77b4", marker="o", markersize=4)
    ax.plot(betas, sosr_vals, label="SOSR-TTA (Ours)", color="#2ca02c", marker="s", markersize=4)
    
    ax.set_xscale("log")
    ax.set_xlabel(r"Regularization Coefficient $\beta$")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(betas)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", ls="-")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.8)

plt.tight_layout()
plt.savefig("./beta_sensitivity.pdf", dpi=300)
plt.close()
print("Saved beta_sensitivity.pdf (compact version)")
