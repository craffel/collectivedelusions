import matplotlib.pyplot as plt
import numpy as np

# Learning rate grids
lr_sgd = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
lr_adam = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])

# 1. Alternating Stream Data
alt_uniform = [55.27, 56.59, 57.03, 56.98, 55.66, 54.20]
alt_lfwa = [46.24, 42.24, 39.99, 40.97, 39.65, 33.69]
alt_adasnr = [55.18, 55.91, 56.45, 56.64, 56.01, 56.69]
alt_ours = [56.10, 57.62, 59.57, 55.96, 52.39, 50.29]

# 2. Block-Sequential Stream Data
seq_uniform = [55.71, 56.88, 57.52, 58.25, 56.40, 52.05]
seq_lfwa = [48.63, 58.64, 54.15, 55.08, 48.49, 41.50]
seq_adasnr = [55.52, 55.96, 56.93, 57.23, 57.37, 55.96]
seq_ours = [55.96, 57.28, 58.11, 49.37, 50.63, 51.86]

# Set style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')

# Plot Alternating Stream
plt.figure(figsize=(6, 4))
plt.plot(lr_sgd, alt_uniform, marker='o', color='darkorange', linewidth=2, label='Uniform TTA (SGD)')
plt.plot(lr_sgd, alt_lfwa, marker='s', color='red', linewidth=2, label='LFWA TTA (SGD)')
plt.plot(lr_sgd, alt_adasnr, marker='^', color='green', linewidth=2, label='AdaSNR TTA (SGD-Standard)')
plt.plot(lr_adam, alt_ours, marker='*', color='blue', linewidth=2.5, markersize=8, label='AdaSNR-Adam-TC (Ours)')
plt.axhline(y=54.93, color='gray', linestyle='--', label='Static Merging')
plt.xscale('log')
plt.xlabel('Base Learning Rate (Log Scale)')
plt.ylabel('Average Accuracy (%)')
plt.title('Sweep on Alternating Stream')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(fontsize=8, loc='lower left')
plt.tight_layout()
plt.savefig('template/fig_sweeps_alt.png', dpi=300)
plt.close()

# Plot Block-Sequential Stream
plt.figure(figsize=(6, 4))
plt.plot(lr_sgd, seq_uniform, marker='o', color='darkorange', linewidth=2, label='Uniform TTA (SGD)')
plt.plot(lr_sgd, seq_lfwa, marker='s', color='red', linewidth=2, label='LFWA TTA (SGD)')
plt.plot(lr_sgd, seq_adasnr, marker='^', color='green', linewidth=2, label='AdaSNR TTA (SGD-Standard)')
plt.plot(lr_adam, seq_ours, marker='*', color='blue', linewidth=2.5, markersize=8, label='AdaSNR-Adam-TC (Ours)')
plt.axhline(y=54.93, color='gray', linestyle='--', label='Static Merging')
plt.xscale('log')
plt.xlabel('Base Learning Rate (Log Scale)')
plt.ylabel('Average Accuracy (%)')
plt.title('Sweep on Block-Sequential Stream')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(fontsize=8, loc='lower left')
plt.tight_layout()
plt.savefig('template/fig_sweeps_seq.png', dpi=300)
plt.close()

print("Generated professional plots and saved to template/ folder!")
