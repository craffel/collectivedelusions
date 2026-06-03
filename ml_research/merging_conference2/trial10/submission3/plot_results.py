import matplotlib.pyplot as plt
import numpy as np

# Set style for professional look
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'legend.fontsize': 10,
    'font.family': 'sans-serif'
})

# Create figure with three panels side-by-side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: Multitasking accuracy comparison (FP32)
methods = ['Oracle\nExperts', 'Weight\nAveraging', 'DE-BN\n(32 samples)', 'BW-ADSR\n(temp=-5.0)', 'CW-ADSR\n(temp=-0.5)', 'AS-ADSR\n(Ours, Adaptive)', 'SS-ADSR\n(Ours, Spectral)']
mnist_accs = [99.10, 51.92, 51.91, 74.98, 59.98, 74.95, 72.77]
fmnist_accs = [91.90, 50.61, 53.51, 61.25, 61.28, 61.27, 62.05]
cifar_accs = [79.86, 38.32, 38.55, 56.82, 37.16, 56.82, 55.38]
avg_accs = [90.29, 46.95, 47.99, 64.35, 52.81, 64.35, 63.40]

x = np.arange(len(methods))
width = 0.11

ax1.bar(x - 2.0*width, mnist_accs, width, label='MNIST', color='#4F81BD')
ax1.bar(x - 1.0*width, fmnist_accs, width, label='FashionMNIST', color='#C0504D')
ax1.bar(x, cifar_accs, width, label='CIFAR-10', color='#9BBB59')
ax1.bar(x + 1.0*width, avg_accs, width, label='Average', color='#8064A2', hatch='//')

ax1.set_ylabel('Accuracy (%)')
ax1.set_title('FP32 Multitasking Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=15, ha='right')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')

# Panel B: PTQ Robustness Curve (Average Accuracy)
bits = [32, 8, 6, 4] # represented as bits
wa_ptq = [46.95, 47.24, 45.76, 16.17]
de_bn_ptq = [47.99, 47.78, 47.06, 16.46] # updated from evaluation
bw_adsr_ptq = [64.35, 64.73, 59.71, 16.87]
cw_adsr_ptq = [52.81, 53.04, 51.03, 15.50]
as_adsr_ptq = [64.35, 64.81, 59.55, 16.84]
ss_adsr_ptq = [63.40, 63.75, 59.22, 15.90]

ax2.plot(bits, wa_ptq, 'o-', linewidth=2.0, markersize=7, label='Weight Averaging', color='#C0504D')
ax2.plot(bits, de_bn_ptq, 's--', linewidth=2.0, markersize=7, label='DE-BN (32 samples)', color='#9BBB59')
ax2.plot(bits, bw_adsr_ptq, '^-.', linewidth=2.0, markersize=8, label='BW-ADSR (Ours)', color='#4F81BD')
ax2.plot(bits, cw_adsr_ptq, 'd:', linewidth=2.0, markersize=8, label='CW-ADSR (Ours)', color='#8064A2')
ax2.plot(bits, as_adsr_ptq, 'p-', linewidth=3.0, markersize=10, label='AS-ADSR (Ours, Adaptive)', color='#FFC000')
ax2.plot(bits, ss_adsr_ptq, 'x-', linewidth=2.5, markersize=8, label='SS-ADSR (Ours, Spectral)', color='#008080')

ax2.set_xlabel('Quantization Bits')
ax2.set_ylabel('Average Accuracy (%)')
ax2.set_title('Post-Training Quantization (PTQ)')
ax2.set_xticks([32, 8, 6, 4])
ax2.set_xticklabels(['FP32', '8-bit', '6-bit', '4-bit'])
ax2.set_xlim(34, 2) # invert x axis to show decreasing bits
ax2.set_ylim(0, 100)
ax2.grid(linestyle='--', alpha=0.7)
ax2.legend(loc='lower left')

# Panel C: Input Gaussian Noise Robustness (Average Accuracy)
noise_levels = [0.0, 0.1, 0.2, 0.3]
wa_noise = [46.95, 44.87, 39.10, 33.77] # updated from evaluation
de_bn_noise = [47.99, 46.89, 41.63, 36.17] # updated from evaluation
bw_adsr_noise = [64.35, 57.94, 43.03, 28.62] # updated from evaluation
cw_adsr_noise = [52.81, 53.66, 48.03, 41.26] # updated from evaluation
as_adsr_noise = [64.35, 57.86, 42.89, 28.74] # updated from evaluation
ss_adsr_noise = [63.40, 57.32, 40.61, 27.48]

ax3.plot(noise_levels, wa_noise, 'o-', linewidth=2.0, markersize=7, label='Weight Averaging', color='#C0504D')
ax3.plot(noise_levels, de_bn_noise, 's--', linewidth=2.0, markersize=7, label='DE-BN (32 samples)', color='#9BBB59')
ax3.plot(noise_levels, bw_adsr_noise, '^-.', linewidth=2.0, markersize=8, label='BW-ADSR (Ours)', color='#4F81BD')
ax3.plot(noise_levels, cw_adsr_noise, 'd:', linewidth=2.0, markersize=8, label='CW-ADSR (Ours)', color='#8064A2')
ax3.plot(noise_levels, as_adsr_noise, 'p-', linewidth=3.0, markersize=10, label='AS-ADSR (Ours, Adaptive)', color='#FFC000')
ax3.plot(noise_levels, ss_adsr_noise, 'x-', linewidth=2.5, markersize=8, label='SS-ADSR (Ours, Spectral)', color='#008080')

ax3.set_xlabel(r'Gaussian Noise std ($\sigma$)')
ax3.set_ylabel('Average Accuracy (%)')
ax3.set_title('Environmental Noise Resilience')
ax3.set_xticks(noise_levels)
ax3.set_xlim(-0.02, 0.32)
ax3.set_ylim(20, 100)
ax3.grid(linestyle='--', alpha=0.7)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig('results_plot.png', dpi=300)
print("Successfully generated three-panel results_plot.png.")
