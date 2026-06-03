import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style using matplotlib defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# ----------------------------------------------------------------------
# FIGURE 1: Clean vs OOD Average Robustness Pareto Frontier
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=300)

# Colors and markers for the merging paradigms and strategies
merges = {
    "WA": {"color": "#1f77b4", "marker": "o", "label": "Weight Averaging (WA)"},
    "TA": {"color": "#ff7f0e", "marker": "s", "label": "Task Arithmetic (TA)"},
    "TIES": {"color": "#2ca02c", "marker": "^", "label": "TIES-Merging"},
    "DARE": {"color": "#d62728", "marker": "d", "label": "DARE-Merging"},
}

strategies = {
    "None": {"marker": "x", "size": 60, "alpha": 0.5},
    "BN-Calib": {"marker": "+", "size": 80, "alpha": 0.7},
    "SLR-WBC": {"marker": "p", "size": 80, "alpha": 0.7},
    "MSPR": {"marker": "v", "size": 80, "alpha": 0.7},
    "REC-SVD": {"marker": "h", "size": 80, "alpha": 0.7},
    "REC-Routing": {"marker": "*", "size": 150, "alpha": 1.0}, # Star for ours!
}

# Performance data: (Clean Avg, OOD Avg)
data = {
    "WA": {
        "None": (46.43, 28.98),
        "BN-Calib": (72.20, 37.29),
        "SLR-WBC": (72.97, 39.47),
        "MSPR": (68.83, 33.50),
        "REC-SVD": (57.07, 36.73),
        "REC-Routing": (71.53, 39.95),
    },
    "TA": {
        "None": (9.33, 9.33),
        "BN-Calib": (76.37, 40.43),
        "SLR-WBC": (76.13, 41.94),
        "MSPR": (72.97, 36.23),
        "REC-SVD": (61.60, 39.36),
        "REC-Routing": (75.83, 43.19),
    },
    "TIES": {
        "None": (21.97, 17.36),
        "BN-Calib": (79.07, 43.39),
        "SLR-WBC": (78.23, 44.69),
        "MSPR": (75.53, 39.16),
        "REC-SVD": (60.17, 41.38),
        "REC-Routing": (78.03, 46.44),
    },
    "DARE": {
        "None": (9.33, 9.33),
        "BN-Calib": (71.80, 36.84),
        "SLR-WBC": (72.67, 38.99),
        "MSPR": (68.40, 33.12),
        "REC-SVD": (52.23, 35.02),
        "REC-Routing": (71.17, 39.75),
    }
}

# Plot the points
for m_name, m_info in merges.items():
    xs = []
    ys = []
    for s_name, s_info in strategies.items():
        x, y = data[m_name][s_name]
        xs.append(x)
        ys.append(y)
        # Choose specific marker representation
        marker = s_info["marker"]
        size = s_info["size"]
        alpha = s_info["alpha"]
        
        # Plot individual strategy point
        ax.scatter(x, y, color=m_info["color"], marker=marker, s=size, alpha=alpha, edgecolors='black', linewidths=0.5)
        
        # Add labels to ours
        if s_name == "REC-Routing":
            ax.annotate(f"Ours ({m_name})", (x, y), textcoords="offset points", xytext=(4, 5), ha='left', weight='bold', fontsize=8)

# Dummy plots for legend
for m_name, m_info in merges.items():
    ax.scatter([], [], color=m_info["color"], marker='o', s=60, label=m_info["label"], edgecolors='black')

for s_name, s_info in strategies.items():
    label_name = f"REC-{s_name.split('-')[1]}" if s_name.startswith("REC") else s_name
    if s_name == "REC-Routing":
        label_name = "REC-Routing (Ours)"
    elif s_name == "REC-SVD":
        label_name = "REC-SVD (Ours)"
    ax.scatter([], [], color='gray', marker=s_info["marker"], s=s_info["size"], label=label_name)

ax.set_xlabel("Clean Average Accuracy (%)", weight='bold')
ax.set_ylabel("OOD Average Accuracy (%)", weight='bold')
ax.set_title("Pareto Robustness Frontier of Model Merging Calibration", weight='bold', pad=10)
ax.grid(True, linestyle="--", alpha=0.5)

# Place legend outside or inside beautifully
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0., fontsize=8)

plt.tight_layout()
plt.savefig("robustness_pareto.pdf", format="pdf", bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# FIGURE 2: Task-Vector Scaling Sweep
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.1), dpi=300)

lambdas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# Data for OOD Avg
none_ood = np.array([14.78, 21.92, 28.66, 9.33, 9.33, 9.33, 9.33, 9.33])
bn_calib_ood = np.array([18.26, 28.13, 35.51, 40.53, 44.05, 46.72, 48.23, 49.32])
slr_wbc_ood = np.array([23.42, 31.76, 38.12, 41.98, 45.47, 47.78, 49.28, 50.80])
rec_routing_ood = np.array([18.64, 29.01, 37.39, 43.48, 47.29, 49.30, 51.29, 51.36])

ax.plot(lambdas, none_ood, label="None (No Calib)", color="#7f7f7f", linestyle="--", marker="x", linewidth=1.5, markersize=6)
ax.plot(lambdas, bn_calib_ood, label="BN-Calib", color="#1f77b4", linestyle="-", marker="+", linewidth=1.5, markersize=8)
ax.plot(lambdas, slr_wbc_ood, label="SLR-WBC", color="#2ca02c", linestyle="-", marker="p", linewidth=1.5, markersize=6)
ax.plot(lambdas, rec_routing_ood, label="REC-Routing (Ours)", color="#d62728", linestyle="-", marker="*", linewidth=2.0, markersize=10)

ax.set_xlabel(r"Task Arithmetic Scaling Coefficient ($\lambda$)", weight='bold')
ax.set_ylabel("OOD Average Accuracy (%)", weight='bold')
ax.set_title("OOD Accuracy vs. Scaling Scale $\lambda$ under TA", weight='bold', pad=10)
ax.set_xticks(lambdas)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right", fontsize=9)

# Shade the catastrophic collapse region
ax.axvspan(0.38, 0.82, color='red', alpha=0.08, label="Catastrophic Phase Collapse")
ax.text(0.6, 12, "Collapse Region\n(Uncalibrated)", color='#a00000', weight='bold', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("lambda_sweep.pdf", format="pdf", bbox_inches='tight')
plt.close()

print("Figures successfully generated!")
