import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results
with open("calibration_size_results.json", "r") as f:
    results = json.load(f)

cal_sizes = [8, 16, 32, 64, 128]
K_values = [2, 5, 10]

# Set style
plt.style.use('seaborn-v0_8-paper')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

colors = {2: '#1f77b4', 5: '#ff7f0e', 10: '#2ca02c'}
markers = {'WA_BN': 'o', 'TA_BN': 's'}
linestyles = {'WA_BN': '-', 'TA_BN': '--'}

# Plot Oracle Gated Accuracy
for K in K_values:
    # WA_BN
    wa_oracle = [res["oracle_acc"] for res in results["WA_BN"][str(K)]]
    ax1.plot(cal_sizes, wa_oracle, color=colors[K], marker=markers['WA_BN'], 
             linestyle=linestyles['WA_BN'], label=f"WA+BN (K={K})")
    
    # TA_BN
    ta_oracle = [res["oracle_acc"] for res in results["TA_BN"][str(K)]]
    ax1.plot(cal_sizes, ta_oracle, color=colors[K], marker=markers['TA_BN'], 
             linestyle=linestyles['TA_BN'], label=f"TA+BN (K={K})")

ax1.set_xscale('log')
ax1.set_xticks(cal_sizes)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel("Calibration Set Size (Samples per Task)", fontsize=10)
ax1.set_ylabel("Oracle Gated Accuracy (%)", fontsize=10)
ax1.set_title("Oracle Gated Accuracy vs. Calibration Size", fontsize=11, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot Routing Accuracy
for K in K_values:
    # WA_BN
    wa_routing = [res["routing_acc"] for res in results["WA_BN"][str(K)]]
    ax2.plot(cal_sizes, wa_routing, color=colors[K], marker=markers['WA_BN'], 
             linestyle=linestyles['WA_BN'], label=f"WA+BN (K={K})")
    
    # TA_BN
    ta_routing = [res["routing_acc"] for res in results["TA_BN"][str(K)]]
    ax2.plot(cal_sizes, ta_routing, color=colors[K], marker=markers['TA_BN'], 
             linestyle=linestyles['TA_BN'], label=f"TA+BN (K={K})")

# Add random guess baseline for K=10
ax2.axhline(10.0, color='gray', linestyle=':', alpha=0.7, label="Random Guess (K=10)")
ax2.axhline(20.0, color='gray', linestyle='-.', alpha=0.7, label="Random Guess (K=5)")
ax2.axhline(50.0, color='gray', linestyle='--', alpha=0.7, label="Random Guess (K=2)")

ax2.set_xscale('log')
ax2.set_xticks(cal_sizes)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.set_xlabel("Calibration Set Size (Samples per Task)", fontsize=10)
ax2.set_ylabel("MSPR Routing Accuracy (%)", fontsize=10)
ax2.set_title("MSPR Gating Accuracy vs. Calibration Size", fontsize=11, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)

# Handle single unified legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=9)

plt.tight_layout()
plt.savefig("calibration_size_ablation.png", bbox_inches='tight')
print("Successfully generated calibration_size_ablation.png!")
