import matplotlib.pyplot as plt
import numpy as np

# Set style for professional look
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

results = [
    # WA, N=16
    {"Mode": "WA", "Method": "Uncalibrated", "N": 16, "Average": 22.94},
    {"Mode": "WA", "Method": "SPTAAC", "N": 16, "Average": 24.72},
    {"Mode": "WA", "Method": "WRSA", "N": 16, "Average": 22.59},
    {"Mode": "WA", "Method": "NRA", "N": 16, "Average": 20.26},
    # TA, N=16
    {"Mode": "TA", "Method": "Uncalibrated", "N": 16, "Average": 24.49},
    {"Mode": "TA", "Method": "SPTAAC", "N": 16, "Average": 25.09},
    {"Mode": "TA", "Method": "WRSA", "N": 16, "Average": 24.75},
    {"Mode": "TA", "Method": "NRA", "N": 16, "Average": 21.27},
    # WA, N=64
    {"Mode": "WA", "Method": "Uncalibrated", "N": 64, "Average": 22.82},
    {"Mode": "WA", "Method": "SPTAAC", "N": 64, "Average": 25.53},
    {"Mode": "WA", "Method": "WRSA", "N": 64, "Average": 23.02},
    {"Mode": "WA", "Method": "NRA", "N": 64, "Average": 25.61},
    # TA, N=64
    {"Mode": "TA", "Method": "Uncalibrated", "N": 64, "Average": 24.75},
    {"Mode": "TA", "Method": "SPTAAC", "N": 64, "Average": 25.47},
    {"Mode": "TA", "Method": "WRSA", "N": 64, "Average": 24.99},
    {"Mode": "TA", "Method": "NRA", "N": 64, "Average": 27.05},
    # WA, N=128
    {"Mode": "WA", "Method": "Uncalibrated", "N": 128, "Average": 22.90},
    {"Mode": "WA", "Method": "SPTAAC", "N": 128, "Average": 26.46},
    {"Mode": "WA", "Method": "WRSA", "N": 128, "Average": 22.88},
    {"Mode": "WA", "Method": "NRA", "N": 128, "Average": 26.56},
    # TA, N=128
    {"Mode": "TA", "Method": "Uncalibrated", "N": 128, "Average": 24.73},
    {"Mode": "TA", "Method": "SPTAAC", "N": 128, "Average": 26.01},
    {"Mode": "TA", "Method": "WRSA", "N": 128, "Average": 24.80},
    {"Mode": "TA", "Method": "NRA", "N": 128, "Average": 26.64},
]

cal_sizes = [16, 64, 128]
methods = ["Uncalibrated", "SPTAAC", "WRSA", "NRA (Ours)"]
colors = {
    "Uncalibrated": "#7f7f7f",  # Gray
    "SPTAAC": "#1f77b4",        # Blue
    "WRSA": "#ff7f0e",          # Orange
    "NRA (Ours)": "#d62728"     # Red (Highlighted)
}
markers = {
    "Uncalibrated": "x",
    "SPTAAC": "s",
    "WRSA": "^",
    "NRA (Ours)": "o"
}

# 1. Plot Weight Averaging (WA)
plt.figure(figsize=(6, 4.5))
for method in methods:
    y_vals = []
    m_key = "NRA" if "Ours" in method else method
    for size in cal_sizes:
        match = [r for r in results if r["Mode"] == "WA" and r["Method"] == m_key and r["N"] == size]
        if match:
            y_vals.append(match[0]["Average"])
    plt.plot(cal_sizes, y_vals, marker=markers[method], color=colors[method], linewidth=2, markersize=8, label=method)

plt.xlabel("Calibration Dataset Size ($N$ per task)", fontsize=11)
plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=11)
plt.title("Weight Averaging (WA) Calibration", fontsize=12, fontweight='bold')
plt.xticks(cal_sizes)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(frameon=True, facecolor="white", edgecolor="none")
plt.tight_layout()
plt.savefig("calibration_size_comparison_wa.png", dpi=300)
plt.close()
print("Saved calibration_size_comparison_wa.png")

# 2. Plot Task Arithmetic (TA)
plt.figure(figsize=(6, 4.5))
for method in methods:
    y_vals = []
    m_key = "NRA" if "Ours" in method else method
    for size in cal_sizes:
        match = [r for r in results if r["Mode"] == "TA" and r["Method"] == m_key and r["N"] == size]
        if match:
            y_vals.append(match[0]["Average"])
    plt.plot(cal_sizes, y_vals, marker=markers[method], color=colors[method], linewidth=2, markersize=8, label=method)

plt.xlabel("Calibration Dataset Size ($N$ per task)", fontsize=11)
plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=11)
plt.title("Task Arithmetic (TA) Calibration", fontsize=12, fontweight='bold')
plt.xticks(cal_sizes)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(frameon=True, facecolor="white", edgecolor="none")
plt.tight_layout()
plt.savefig("calibration_size_comparison_ta.png", dpi=300)
plt.close()
print("Saved calibration_size_comparison_ta.png")
