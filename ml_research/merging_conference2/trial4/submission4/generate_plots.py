import matplotlib.pyplot as plt
import numpy as np

# Set style for professional publication-quality figures
plt.rcParams.update({
    "text.usetex": False,  # set to False since we don't need LaTeX engine for standard figures
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "savefig.bbox": "tight",
    "savefig.dpi": 300
})

# Color palette (minimalist and professional)
colors = {
    "uncal_wa": "#808080",      # gray
    "sp_taac": "#1f77b4",       # blue
    "taac": "#aec7e8",          # light blue
    "fwmm_head": "#ff7f0e",     # orange
    "momo_orig": "#ffbb78",     # light orange
    "momo_shrink_shift": "#d62728", # red
    "momo_shrink_noshift": "#2ca02c", # green
    "momo_cf_pha": "#8c564b",   # brown
    "reda_sft": "#9467bd"       # purple
}

# --- Plot 1: Main Results ---
# N-sizes
N_sizes = [4, 8, 16, 32, 64, 128, 256]

# Exact numbers from results.txt
uncal_wa = 25.15
sp_taac_only = [58.91, 59.31, 59.67, 59.42, 59.47, 59.47, 59.42]
taac_only = [11.21, 10.62, 26.26, 53.56, 59.03, 65.03, 66.13]
fwmm_only = [38.78, 48.23, 54.20, 55.97, 54.90, 56.09, 56.71]
momo_orig = [20.57, 36.37, 53.27, 61.28, 59.76, 61.49, 62.13]
momo_shrink_shift = [58.97, 60.19, 61.19, 62.18, 60.27, 61.56, 62.12]
momo_shrink_noshift = [59.13, 60.42, 61.38, 61.80, 61.26, 61.55, 61.84]
momo_cf_pha = [56.47, 58.91, 60.85, 63.45, 64.63, 65.94, 67.34]
reda_sft = [28.83, 42.86, 52.71, 62.63, 66.13, 67.67, 70.29]

fig, ax = plt.subplots(figsize=(7, 4.2))

# Plot uncalibrated baseline as a horizontal line
ax.axhline(y=uncal_wa, color=colors["uncal_wa"], linestyle="--", alpha=0.8, label="Uncalibrated WA (25.15%)")

# Plot other lines
ax.plot(N_sizes, sp_taac_only, marker="o", color=colors["sp_taac"], label="SP-TAAC Only", linewidth=2)
ax.plot(N_sizes, taac_only, marker="s", color=colors["taac"], label="TAAC Only", alpha=0.7)
ax.plot(N_sizes, fwmm_only, marker="v", color=colors["fwmm_head"], label="FWMM Only (Head)", alpha=0.7)
ax.plot(N_sizes, momo_orig, marker="p", color=colors["momo_orig"], label="MOMO-Merge (Orig)", alpha=0.7)
ax.plot(N_sizes, momo_shrink_noshift, marker="*", color=colors["momo_shrink_noshift"], label="MOMO-Merge (Shrink-NoShift)", linewidth=2.5)
ax.plot(N_sizes, momo_shrink_shift, marker="D", color=colors["momo_shrink_shift"], label="MOMO-Merge (Shrink-Shift)", alpha=0.8)
ax.plot(N_sizes, momo_cf_pha, marker="h", color=colors["momo_cf_pha"], label="MOMO-Merge (CF-PHA)", linewidth=2.2)
ax.plot(N_sizes, reda_sft, marker="x", color=colors["reda_sft"], label="REDA-SFT (Gradient)", linewidth=2)

ax.set_xscale("log", base=2)
ax.set_xticks(N_sizes)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_xlabel("Calibration Sample Size ($N$)")
ax.set_ylabel("Multi-Task Average Accuracy (%)")
ax.set_title("Multi-Task Merging Accuracy vs. Calibration Set Size")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.set_ylim(5, 75)

# Place legend in a clean manner
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", shadow=False, framealpha=0.9)

plt.tight_layout()
plt.savefig("results_plot.pdf")
plt.savefig("results_plot.png", dpi=300)
plt.close()

# --- Plot 2: Ablation of N0 ---
N0_values = [0, 2, 4, 8, 16, 32, 64, 128]

# Data from ablation_results.txt
n4_shift = [20.57, 53.20, 55.25, 57.34, 58.97, 59.52, 59.40, 59.17]
n4_noshift = [20.16, 54.43, 56.76, 58.48, 59.13, 59.23, 59.13, 59.00]
n8_shift = [36.37, 57.52, 58.26, 59.30, 60.19, 60.51, 60.36, 60.02]
n8_noshift = [37.43, 59.23, 59.90, 60.27, 60.42, 60.35, 60.03, 59.70]

fig, ax = plt.subplots(figsize=(6.5, 4))

ax.plot(N0_values, n4_shift, marker="o", color=colors["momo_shrink_shift"], linestyle="-", label="N=4 (Shrink-Shift)")
ax.plot(N0_values, n4_noshift, marker="s", color=colors["momo_shrink_noshift"], linestyle="-", label="N=4 (Shrink-NoShift)")
ax.plot(N0_values, n8_shift, marker="^", color=colors["momo_shrink_shift"], linestyle="--", label="N=8 (Shrink-Shift)", alpha=0.8)
ax.plot(N0_values, n8_noshift, marker="d", color=colors["momo_shrink_noshift"], linestyle="--", label="N=8 (Shrink-NoShift)", alpha=0.8)

# SP-TAAC references as horizontal lines
ax.axhline(y=58.91, color=colors["sp_taac"], linestyle=":", label="SP-TAAC Only (N=4, 58.91%)", alpha=0.6)
ax.axhline(y=59.31, color=colors["sp_taac"], linestyle="-.", label="SP-TAAC Only (N=8, 59.31%)", alpha=0.6)

# Labels and ticks
ax.set_xscale("symlog", base=2)
ax.set_xticks(N0_values)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_xlabel("Prior Strength ($N_0$)")
ax.set_ylabel("Multi-Task Average Accuracy (%)")
ax.set_title("MOMO-Merge Ablation of Prior Strength $N_0$")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.set_ylim(15, 63)

ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", shadow=False, framealpha=0.9)

plt.tight_layout()
plt.savefig("ablation_plot.pdf")
plt.savefig("ablation_plot.png", dpi=300)
plt.close()

print("Plots successfully generated and saved as results_plot.pdf/png and ablation_plot.pdf/png!")
