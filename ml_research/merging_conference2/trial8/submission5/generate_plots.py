import json
import matplotlib.pyplot as plt
import numpy as np

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.3,
})

# Load results
with open("sweep_results.json", "r") as f:
    results = json.load(f)

# Filter for Task Arithmetic lambda=0.5, Clean data
cal_methods = ['none', 'u-ipr', 'hns', 'qr-ipr']
cal_labels = ['Uncalibrated', 'U-IPR (Ours/S9)', 'HNS (Ours/S7)', 'QR-IPR (Proposed)']

fp32_clean_accs = []
int8_clean_accs = []

for cal in cal_methods:
    # FP32
    r_fp32 = [x for x in results if x['merge_type'] == 'ta' and x['lambda'] == 0.5 and x['calibration'] == cal and x['quantization_bits'] == 'FP32' and x['corruption'] == 'clean'][0]
    fp32_clean_accs.append(r_fp32['avg_acc'])
    
    # INT8
    r_int8 = [x for x in results if x['merge_type'] == 'ta' and x['lambda'] == 0.5 and x['calibration'] == cal and x['quantization_bits'] == 8 and x['corruption'] == 'clean'][0]
    int8_clean_accs.append(r_int8['avg_acc'])

# Plot 1: Quantization Sensitivity (Clean FP32 vs INT8)
fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(cal_methods))
width = 0.35

rects1 = ax.bar(x - width/2, fp32_clean_accs, width, label='FP32 (Full Precision)', color='#1f77b4', alpha=0.9, edgecolor='black', linewidth=0.7)
rects2 = ax.bar(x + width/2, int8_clean_accs, width, label='INT8 (8-bit Quantized)', color='#ff7f0e', alpha=0.9, edgecolor='black', linewidth=0.7)

ax.set_ylabel('Average Accuracy (%)')
ax.set_title('Clean Multi-Task Merging Accuracy (Task Arithmetic)')
ax.set_xticks(x)
ax.set_xticklabels(cal_labels)
ax.set_ylim(0, 80)
ax.legend(frameon=True, facecolor='white', edgecolor='none')

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("quantization_robustness.png", dpi=300)
plt.close()

# Filter for Task Arithmetic lambda=0.5, FP32, Corruptions
clean_accs = []
noise_accs = []
blur_accs = []

for cal in cal_methods:
    r_clean = [x for x in results if x['merge_type'] == 'ta' and x['lambda'] == 0.5 and x['calibration'] == cal and x['quantization_bits'] == 'FP32' and x['corruption'] == 'clean'][0]
    clean_accs.append(r_clean['avg_acc'])
    
    r_noise = [x for x in results if x['merge_type'] == 'ta' and x['lambda'] == 0.5 and x['calibration'] == cal and x['quantization_bits'] == 'FP32' and x['corruption'] == 'noise'][0]
    noise_accs.append(r_noise['avg_acc'])
    
    r_blur = [x for x in results if x['merge_type'] == 'ta' and x['lambda'] == 0.5 and x['calibration'] == cal and x['quantization_bits'] == 'FP32' and x['corruption'] == 'blur'][0]
    blur_accs.append(r_blur['avg_acc'])

# Plot 2: Robustness Under Corruptions (FP32)
fig, ax = plt.subplots(figsize=(7.5, 3.5))
x = np.arange(len(cal_methods))
width = 0.25

rects1 = ax.bar(x - width, clean_accs, width, label='Clean', color='#2ca02c', alpha=0.9, edgecolor='black', linewidth=0.7)
rects2 = ax.bar(x, noise_accs, width, label='Gaussian Noise (std=0.1)', color='#d62728', alpha=0.9, edgecolor='black', linewidth=0.7)
rects3 = ax.bar(x + width, blur_accs, width, label='Gaussian Blur (k=3)', color='#9467bd', alpha=0.9, edgecolor='black', linewidth=0.7)

ax.set_ylabel('Average Accuracy (%)')
ax.set_title('Robustness Under Environmental Corruptions (FP32)')
ax.set_xticks(x)
ax.set_xticklabels(cal_labels)
ax.set_ylim(0, 80)
ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig("noise_robustness.png", dpi=300)
plt.close()

print("Figures saved successfully as quantization_robustness.png and noise_robustness.png!")
