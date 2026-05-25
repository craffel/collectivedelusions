import os
import matplotlib.pyplot as plt
import numpy as np

# Categories for the plot
categories = ['Clean', 'Noise', 'Blur', 'Contrast', 'Rotation', 'OOD Avg']

# Read from experimental_results.txt to be dynamically robust
data = {}
label_map = {
    'Task Arithmetic': 'Task Arithmetic (TA)',
    'AdaMerging': 'AdaMerging',
    'SyMerge': 'SyMerge',
    'SAT-SyMerge': 'SAT-SyMerge (Ours, Tensor-Wise)',
    'ASAM-SyMerge': 'ASAM-SyMerge (Ours, Adaptive)'
}

if os.path.exists('experimental_results.txt'):
    print("Reading experimental results from experimental_results.txt...")
    with open('experimental_results.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) == 7:
                name = parts[0]
                display_name = label_map.get(name, name)
                vals = [float(v) for v in parts[1:]]
                data[display_name] = vals
else:
    print("Warning: experimental_results.txt not found. Falling back to static data.")
    data = {
        'Task Arithmetic (TA)': [46.36, 35.37, 25.59, 10.70, 22.99, 23.66],
        'AdaMerging': [48.31, 33.19, 25.60, 10.89, 22.84, 23.13],
        'SyMerge': [70.16, 30.04, 38.58, 10.54, 31.64, 27.70],
        'SAT-SyMerge (Ours, Tensor-Wise)': [69.27, 29.22, 37.64, 10.52, 31.98, 27.34],
        'ASAM-SyMerge (Ours, Adaptive)': [70.14, 30.04, 38.55, 10.54, 31.71, 27.71]
    }

methods = list(data.keys())

# Plot settings
x = np.arange(len(categories))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 7))

# 5 distinct, professional colors: Gray, Blue, Red, Purple, Green
colors = ['#95a5a6', '#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

# Center 5 bars by shifting them relative to the index
for i, method in enumerate(methods):
    color = colors[i % len(colors)]
    rects = ax.bar(x + (i - 2.0) * width, data[method], width, label=method, color=color, edgecolor='black', linewidth=0.7)

# Labels and title
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Merging Performance Comparison under Domain Shifts & Corruptions', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 85)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(fontsize=10, loc='upper right', frameon=True, shadow=True)

# Add value labels on top of the bars with small offset and angle to avoid collision
for i, method in enumerate(methods):
    for j, val in enumerate(data[method]):
        ax.annotate(f'{val:.1f}',
                    xy=(x[j] + (i - 2.0) * width, val),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig('results_plot.png', dpi=300)
print("Plot successfully updated and saved as results_plot.png!")
