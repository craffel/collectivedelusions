import json
import matplotlib.pyplot as plt
import os

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

files = {
    "Standard LoRA (Baseline)": "results_standard.json",
    "SAM Only (Flatness)": "results_sam.json",
    "ISR Only (Rigid)": "results_isr.json",
    "ISR Only (Soft SOSR)": "results_isr_soft.json",
    "SATA-LR (Rigid)": "results_sata_lr.json",
    "SATA-LR-Soft (SAM + SOSR)": "results_sata_lr_soft.json"
}

colors = {
    "Standard LoRA (Baseline)": "#7F8C8D",       # Gray
    "SAM Only (Flatness)": "#E67E22",            # Orange
    "ISR Only (Rigid)": "#9B59B6",               # Purple
    "ISR Only (Soft SOSR)": "#8E44AD",           # Deep Purple
    "SATA-LR (Rigid)": "#3498DB",                # Blue
    "SATA-LR-Soft (SAM + SOSR)": "#2980B9"       # Deep Blue
}

markers = {
    "Standard LoRA (Baseline)": "o",
    "SAM Only (Flatness)": "s",
    "ISR Only (Rigid)": "^",
    "ISR Only (Soft SOSR)": "v",
    "SATA-LR (Rigid)": "d",
    "SATA-LR-Soft (SAM + SOSR)": "D"
}

fig, ax = plt.subplots(figsize=(7, 4.5))

for name, filename in files.items():
    if not os.path.exists(filename):
        print(f"File {filename} not found, skipping...")
        continue
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # We plot SVDM as it is the best performing merging method
    svdm_data = data.get("SVDM", [])
    if not svdm_data:
        continue
    
    lambdas = [item["lambda"] for item in svdm_data]
    avg_accs = [item["avg_acc"] for item in svdm_data]
    
    ax.plot(lambdas, avg_accs, label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=6)

ax.set_xlabel(r"Interpolation Factor $\lambda$ (CIFAR-10 $\leftarrow$ SVHN)", fontsize=12)
ax.set_ylabel("Multi-Task Average Accuracy (%)", fontsize=12)
ax.set_title("SVD-based Merging (SVDM) Performance across Configurations", fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(45, 100)
ax.set_xticks([i * 0.1 for i in range(11)])
ax.set_yticks(range(50, 101, 5))

ax.legend(loc='lower center', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#BDC3C7', ncol=2)
plt.tight_layout()

plt.savefig("lambda_sweep_svdm.pdf", bbox_inches='tight', dpi=300)
plt.savefig("lambda_sweep_svdm.png", bbox_inches='tight', dpi=300)
print("Successfully generated lambda_sweep_svdm.pdf and lambda_sweep_svdm.png!")
