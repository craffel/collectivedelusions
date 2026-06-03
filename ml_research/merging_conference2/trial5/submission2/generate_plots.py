import matplotlib.pyplot as plt
import numpy as np

# Set up beautiful plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

# Colors matching the method groups
colors = {
    'WA': '#2c3e50',       # Dark blue-grey
    'SP-TAAC': '#2980b9',  # Bright blue
    'TAAC': '#e67e22',     # Orange
    'ZIO-CF': '#27ae60',   # Green
    'L-FDSA': '#9b59b6',   # Purple
    'C-FDSA': '#c0392b',   # Dark Red
    'SRAC': '#f1c40f'      # Yellow
}

# ==========================================
# Plot 1: Latency vs. Accuracy (Standalone)
# ==========================================
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

# Latency (ms) and Standalone Accuracy (%)
methods = {
    'Uncalibrated WA': {'latency': 70.5334, 'accuracy': 41.27, 'color': colors['WA'], 'marker': 'o', 'offset': (-15, 10)},
    'SP-TAAC': {'latency': 71.8232, 'accuracy': 43.48, 'color': colors['SP-TAAC'], 'marker': 's', 'offset': (10, -5)},
    'SRAC (Dynamic)': {'latency': 76.3852, 'accuracy': 42.31, 'color': colors['SRAC'], 'marker': '^', 'offset': (10, 5)},
    'TAAC (Hooked)': {'latency': 85.7282, 'accuracy': 19.21, 'color': colors['TAAC'], 'marker': 'D', 'offset': (10, -5)},
    'ZIO-CF (Fused)': {'latency': 70.5334, 'accuracy': 19.21, 'color': colors['ZIO-CF'], 'marker': 'v', 'offset': (-55, -15)},
    'ZIO-CF (Hooked)': {'latency': 94.7756, 'accuracy': 19.21, 'color': colors['ZIO-CF'], 'marker': 'x', 'offset': (10, -5)},
    'L-FDSA (Fourier Global)': {'latency': 234.1219, 'accuracy': 33.30, 'color': colors['L-FDSA'], 'marker': 'h', 'offset': (-130, -15)},
    'C-FDSA (Fourier Channel)': {'latency': 146.4849, 'accuracy': 12.16, 'color': colors['C-FDSA'], 'marker': 'p', 'offset': (-85, 12)}
}

for name, data in methods.items():
    ec = 'black' if data['marker'] != 'x' else None
    ax.scatter(data['latency'], data['accuracy'], color=data['color'], marker=data['marker'], s=100, edgecolors=ec, linewidths=0.8, zorder=5, label=name)
    ax.annotate(name, (data['latency'], data['accuracy']), textcoords="offset points", xytext=data['offset'], ha='left', va='center', fontsize=8.5, fontweight='semibold', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

# Draw arrow from ZIO-CF Hooked to ZIO-CF Fused to show compilation benefits
ax.annotate("", xy=(70.5334, 19.21), xytext=(94.7756, 19.21), arrowprops=dict(arrowstyle="->", color=colors['ZIO-CF'], lw=1.5, ls="--"))
ax.text(82.6, 21.0, "Mathematical\nFusion", color=colors['ZIO-CF'], fontsize=8, ha='center', fontweight='bold')

ax.set_xlabel("Inference Latency (ms per batch, batch size = 128)", fontweight='bold')
ax.set_ylabel("Average Multitask Test Accuracy (%)", fontweight='bold')
ax.set_title("Latency-Accuracy Trade-off in Model Merging Calibration", fontweight='bold', pad=15)
ax.set_xlim(55, 260)
ax.set_ylim(5, 55)
ax.grid(True, linestyle='--', alpha=0.5)

# Highlight Pareto Frontier area
# SP-TAAC is highest accuracy at extremely low latency, WA is slightly lower accuracy at same base latency.
# ZIO-CF has same base latency but suffers from Sparsity Trap.
rect = plt.Rectangle((60, 40), 20, 6, facecolor='green', alpha=0.1, edgecolor='green', linestyle=':', label='Pareto Optimal Region')
ax.add_patch(rect)

plt.tight_layout()
plt.savefig("latency_vs_accuracy.png", bbox_inches='tight')
plt.savefig("latency_vs_accuracy.pdf", bbox_inches='tight')
plt.close()

# ==========================================
# Plot 2: Robustness to Calibration Scarcity
# ==========================================
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

N_samples = [4, 16, 64, 128]

wa_sft = [28.73, 24.57, 40.60, 43.05]
sp_taac_sft = [36.19, 44.24, 44.93, 45.15]
zio_cf_sft = [17.27, 17.86, 17.93, 17.49]
l_fdsa_sft = [26.97, 28.76, 36.30, 36.69]

ax.plot(N_samples, sp_taac_sft, marker='s', markersize=8, color=colors['SP-TAAC'], linewidth=2.5, label='SP-TAAC + SFT (Global Spatial)', zorder=5)
ax.plot(N_samples, wa_sft, marker='o', markersize=8, color=colors['WA'], linewidth=2, linestyle='--', label='Uncalibrated WA + SFT (Base)', zorder=4)
ax.plot(N_samples, l_fdsa_sft, marker='h', markersize=8, color=colors['L-FDSA'], linewidth=2, linestyle='-.', label='L-FDSA + SFT (Fourier Global)', zorder=3)
ax.plot(N_samples, zio_cf_sft, marker='v', markersize=8, color=colors['ZIO-CF'], linewidth=2, linestyle=':', label='ZIO-CF + SFT (Sparsity Trap)', zorder=2)

ax.set_xscale('log')
ax.set_xticks(N_samples)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("Calibration Dataset Size $N$ (samples per task, log-scale)", fontweight='bold')
ax.set_ylabel("Average Multitask Test Accuracy (%)", fontweight='bold')
ax.set_title("Robustness under Calibration Data Scarcity", fontweight='bold', pad=15)
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.set_ylim(10, 50)
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

# Add a text callout for the Sparsity Trap
ax.text(16, 14.5, "The Sparsity Trap\n(Channel-wise collapse)", color=colors['ZIO-CF'], fontsize=9, fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.2", fc="#e8f8f5", ec="none", alpha=0.85))
ax.annotate("", xy=(16, 17.2), xytext=(16, 15.5), arrowprops=dict(arrowstyle="->", color=colors['ZIO-CF'], lw=1))

# Add text callout for SP-TAAC robustness
ax.text(8, 41.0, "SP-TAAC\nRobustness", color=colors['SP-TAAC'], fontsize=9, fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.2", fc="#ebf5fb", ec="none", alpha=0.85))

plt.tight_layout()
plt.savefig("robustness_scarcity.png", bbox_inches='tight')
plt.savefig("robustness_scarcity.pdf", bbox_inches='tight')
plt.close()

print("Plots generated successfully!")
