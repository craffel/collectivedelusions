import torch
import matplotlib.pyplot as plt
import numpy as np

# Load results
results = torch.load('checkpoints/stream_results.pt', map_location='cpu', weights_only=False)

plt.figure(figsize=(12, 6))

methods = [
    ("Method A", "Fixed TTA + Reset (L2)"),
    ("Method B", "CL W-Fisher + SCTS (L2)"),
    ("Method C", "CL W-Fisher + A-SCTS"),
    ("Method D", "CP-AM (Ours in paper 1)"),
    ("Method E", "BK-AHR (Proposed)")
]

colors = ['grey', 'blue', 'green', 'orange', 'red']
styles = ['--', '-.', ':', '-', '-']

for (m_id, m_label), color, style in zip(methods, colors, styles):
    accs = results[m_id]['accuracies']
    plt.plot(accs, label=m_label, color=color, linestyle=style, linewidth=2)

# Mark boundaries of the stream phases
boundaries = [10, 20, 30, 40]
phases = ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]

for b in boundaries:
    plt.axvline(x=b-0.5, color='black', linestyle=':', alpha=0.5)

# Label phases
plt.text(4.5, 93, "Clean\nMNIST", ha='center', fontsize=9, fontweight='bold')
plt.text(14.5, 93, "Noisy\nMNIST", ha='center', fontsize=9, fontweight='bold')
plt.text(24.5, 93, "Clean\nFashion", ha='center', fontsize=9, fontweight='bold')
plt.text(34.5, 93, "Noisy\nFashion", ha='center', fontsize=9, fontweight='bold')
plt.text(44.5, 93, "Novel\nKMNIST", ha='center', fontsize=9, fontweight='bold')

plt.title("Test-Time Model Merging Accuracy across Streaming Covariate Shifts", fontsize=14, fontweight='bold')
plt.xlabel("Test Stream Batch Index", fontsize=12)
plt.ylabel("Classification Accuracy (%)", fontsize=12)
plt.ylim(0, 105)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='grey')

plt.tight_layout()
plt.savefig('results_plot.png', dpi=300)
print("Saved stream evaluation plot as results_plot.png!")
