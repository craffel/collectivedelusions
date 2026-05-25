import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Parse results from results_summary.txt
results = {}
with open("results_summary.txt", "r") as f:
    for line in f:
        if line.startswith("alternating") or line.startswith("sequential"):
            parts = [p.strip().replace('%', '') for p in line.split("|")]
            stream_type = parts[0]
            # Convert values to float
            vals = [float(x) for x in parts[1:]]
            results[stream_type] = vals

methods = ['Static', 'Std TTA', 'L2\n(G=1)', 'L2\n(G=10)', 'L2\n(G=100)', 'EWC\n(G=1)', 'EWC\n(G=10)', 'EWC\n(G=100)']
colors = ['#2b5c8f', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#1b9e77', '#d95f02', '#7570b3']

# Plot Alternating Stream
alt_accs = results['alternating']
bars1 = ax1.bar(methods, alt_accs, color=colors, width=0.6, edgecolor='black', linewidth=0.7)
ax1.set_title('Alternating Stream (High Frequency)', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Overall Accuracy (%)', fontsize=11)
ax1.set_ylim(0, 100)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot Sequential Stream
seq_accs = results['sequential']
bars2 = ax2.bar(methods, seq_accs, color=colors, width=0.6, edgecolor='black', linewidth=0.7)
ax2.set_title('Sequential Stream (Severe Domain Shift)', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel('Overall Accuracy (%)', fontsize=11)
ax2.set_ylim(0, 100)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.suptitle('Performance of TTA Methods vs. Baselines (Reproducible Seed=42 Streams)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results_plot.png', dpi=300, bbox_inches='tight')
print("Successfully generated results_plot.png!")
