import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

chunk_sizes = []
static_accs = []
std_accs = []
l2_accs = []
ewc_accs = []

with open("switching_frequency_results.txt", "r") as f:
    for line in f:
        if line.strip().startswith("|") and not ":" in line and not "Chunk Size" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6:
                try:
                    m = int(parts[1])
                    static = float(parts[2].replace('%', ''))
                    std = float(parts[3].replace('%', ''))
                    l2 = float(parts[4].replace('%', ''))
                    ewc = float(parts[5].replace('%', ''))
                    
                    chunk_sizes.append(m)
                    static_accs.append(static)
                    std_accs.append(std)
                    l2_accs.append(l2)
                    ewc_accs.append(ewc)
                except ValueError:
                    continue

# Plot curves
ax.plot(chunk_sizes, static_accs, 'k--', label='Static Merged', linewidth=1.5)
ax.plot(chunk_sizes, std_accs, 'o-', color='#d95f02', label='Standard TTA ($\gamma=0$)', linewidth=2.0)
ax.plot(chunk_sizes, l2_accs, 'v:', color='#e7298a', label='L2-TTA ($\gamma=100.0$)', linewidth=2.0)
ax.plot(chunk_sizes, ewc_accs, 's-', color='#1b9e77', label='EWC-TTA ($\gamma=100.0$, Ours)', linewidth=2.5, markersize=7)

ax.set_title('TTA Performance vs. Task-Switching Frequency', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Task Chunk Size $M$ (Number of consecutive batches before task-switch)', fontsize=12)
ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
ax.set_xscale('log') # Log scale is ideal for 1, 2, 5, 10, 25, 50
ax.set_xticks(chunk_sizes)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_ylim(50, 90)
ax.tick_params(axis='both', which='major', labelsize=10)

# Add data annotations for EWC and L2
for m, y_ewc, y_l2 in zip(chunk_sizes, ewc_accs, l2_accs):
    if m in [1, 5, 10, 50]:
        ax.annotate(f'{y_ewc:.1f}%', (m, y_ewc), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, fontweight='bold', color='#1b9e77')
        ax.annotate(f'{y_l2:.1f}%', (m, y_l2), textcoords="offset points", xytext=(0,-13), ha='center', fontsize=9, color='#e7298a')

ax.legend(loc='lower right', frameon=True, fontsize=11)
plt.tight_layout()
plt.savefig('switching_frequency_plot.png', dpi=300, bbox_inches='tight')
print("Successfully generated switching_frequency_plot.png!")
