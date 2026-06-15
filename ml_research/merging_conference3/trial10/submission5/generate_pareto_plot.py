import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

# Data
# Exclude Uniform Static because Jitter is 0 (cannot be represented on log scale easily) and accuracy is low.
data = {
    "SABLE (Stateless)": {"acc": 70.88, "jitter": 298.43, "color": "#7f8c8d", "marker": "o", "group": "Stateless"},
    "ChemMerge (Reset)": {"acc": 70.67, "jitter": 126.57, "color": "#3498db", "marker": "s", "group": "Flat Stateful"},
    "ChemMerge (Coupled)": {"acc": 70.65, "jitter": 51.81, "color": "#2980b9", "marker": "D", "group": "Flat Stateful"},
    "Momentum-Merge (Reset)": {"acc": 69.80, "jitter": 7.74, "color": "#9b59b6", "marker": "v", "group": "Flat Stateful"},
    "Momentum-Merge (Coupled)": {"acc": 88.12, "jitter": 6.01, "color": "#8e44ad", "marker": "^", "group": "Flat Stateful"},
    "UGR (Ours)": {"acc": 92.25, "jitter": 3.68, "color": "#e74c3c", "marker": "*", "size": 180, "group": "UGR (Ours)"},
    "UGR (Hybrid Reset)": {"acc": 92.38, "jitter": 4.41, "color": "#d35400", "marker": "p", "size": 140, "group": "UGR (Ours)"},
    "UGR (Softmax-Free)": {"acc": 87.40, "jitter": 1.50, "color": "#27ae60", "marker": "X", "size": 140, "group": "UGR (Ours)"},
    "UGR (Born Target)": {"acc": 90.67, "jitter": 1.60, "color": "#f1c40f", "marker": "h", "size": 140, "group": "UGR (Ours)"}
}

# Plot groups separately for custom legend
groups = {}
for name, info in data.items():
    g = info["group"]
    if g not in groups:
        groups[g] = {"x": [], "y": [], "color": info["color"], "marker": info["marker"], "names": []}
    groups[g]["x"].append(info["jitter"])
    groups[g]["y"].append(info["acc"])
    groups[g]["names"].append(name)

# Draw points
for g_name, g_info in groups.items():
    if g_name == "UGR (Ours)":
        for x, y, name in zip(g_info["x"], g_info["y"], g_info["names"]):
            marker = data[name]["marker"]
            size = data[name].get("size", 120)
            ax.scatter(x, y, s=size, c=data[name]["color"], marker=marker, 
                       edgecolors='black', linewidths=1.2, zorder=5, label=name)
    else:
        for x, y, name in zip(g_info["x"], g_info["y"], g_info["names"]):
            marker = data[name]["marker"]
            ax.scatter(x, y, s=80, c=data[name]["color"], marker=marker, 
                       edgecolors='black', linewidths=0.8, zorder=4, label=name)

# Draw Pareto Frontier curve for UGR Variants
# The Pareto frontier is formed by UGR (Softmax-Free) [1.50, 87.40], UGR (Born Target) [1.60, 90.67], UGR (Ours) [3.68, 92.25], UGR (Hybrid Reset) [4.41, 92.38]
pareto_x = [1.50, 1.60, 3.68, 4.41]
pareto_y = [87.40, 90.67, 92.25, 92.38]
ax.plot(pareto_x, pareto_y, linestyle='--', color='#e74c3c', alpha=0.8, linewidth=2, zorder=3, label="UGR Pareto Frontier")

# Labels and formatting
ax.set_xscale('log')
ax.set_xlabel(r'Routing Jitter (MSE, $\times 10^{-4}$) $\leftarrow$ [More Stable / Better]', fontsize=11, fontweight='bold', labelpad=8)
ax.set_ylabel('Joint Mean Accuracy (%) $\leftarrow$ [More Accurate / Better]', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('Accuracy-Stability Pareto Frontier on Real-World NLP Stream', fontsize=12, fontweight='bold', pad=12)

# Annotate points with offset to avoid overlap
annotations = {
    "SABLE (Stateless)": (10, -5),
    "ChemMerge (Reset)": (-12, 10),
    "ChemMerge (Coupled)": (-15, -15),
    "Momentum-Merge (Reset)": (10, -8),
    "Momentum-Merge (Coupled)": (10, -5),
    "UGR (Ours)": (12, -5),
    "UGR (Hybrid Reset)": (10, 5),
    "UGR (Softmax-Free)": (-50, -18),
    "UGR (Born Target)": (-60, 10)
}

for name, (x_off, y_off) in annotations.items():
    x = data[name]["jitter"]
    y = data[name]["acc"]
    # Shorter name for annotation
    short_name = name.replace(" (Stateless)", "").replace(" (Reset)", "-R").replace(" (Coupled)", "-C").replace(" (Ours)", "")
    ax.annotate(short_name, (x, y), textcoords="offset points", xytext=(x_off, y_off),
                fontsize=9, fontweight='bold', alpha=0.85)

# Set axes limits to make the plot look clean
ax.set_xlim(0.8, 500)
ax.set_ylim(65, 95)

# Grid styling
ax.grid(True, which="both", linestyle=":", alpha=0.5, color="#bdc3c7")

# Custom Legend to put nicely outside or inside the plot
handles, labels = ax.get_legend_handles_labels()
# Order the legend logically: Pareto curve first, then UGR family, then baselines
order = [labels.index("UGR Pareto Frontier"), 
         labels.index("UGR (Ours)"), 
         labels.index("UGR (Hybrid Reset)"), 
         labels.index("UGR (Born Target)"), 
         labels.index("UGR (Softmax-Free)"),
         labels.index("Momentum-Merge (Coupled)"),
         labels.index("Momentum-Merge (Reset)"),
         labels.index("ChemMerge (Coupled)"),
         labels.index("ChemMerge (Reset)"),
         labels.index("SABLE (Stateless)")]

ordered_handles = [handles[idx] for idx in order]
ordered_labels = [labels[idx] for idx in order]

ax.legend(ordered_handles, ordered_labels, loc='lower left', frameon=True, 
          facecolor='white', edgecolor='#bdc3c7', framealpha=0.9, fontsize=8.5, ncol=2)

# Save high-res plot
plt.tight_layout()
plt.savefig('submission/ablation_pareto.png', dpi=300)
plt.savefig('submission/ablation_pareto.pdf', dpi=300)
print("Pareto plot generated successfully!")
