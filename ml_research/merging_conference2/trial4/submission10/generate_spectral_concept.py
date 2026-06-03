import numpy as np
import matplotlib.pyplot as plt

# Generate simulated frequency spectrum
np.random.seed(42)
freqs = np.linspace(0, 10, 100)

# Original average magnitude
original_magnitude = 10 / (1 + freqs**1.5) + np.random.normal(0, 0.1, 100)
original_magnitude = np.clip(original_magnitude, 0.05, None)

# Collapsed magnitude due to phase mismatch (simulated low-pass filtering)
collapsed_magnitude = original_magnitude * (1.0 - 0.7 * (freqs / 10)**1.2)
collapsed_magnitude = np.clip(collapsed_magnitude, 0.02, None)

# FDSA aligned magnitude (perfectly restored to target)
aligned_magnitude = original_magnitude.copy()

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

# Plot 1: Spectral Collapse (Low-Pass Filter effect)
ax1.plot(freqs, original_magnitude, 'g-', label='Original Experts (Target)', linewidth=2.5)
ax1.plot(freqs, collapsed_magnitude, 'r--', label='Merged Model (Collapsed)', linewidth=2.5)
ax1.fill_between(freqs, collapsed_magnitude, original_magnitude, color='red', alpha=0.15, label='Lost Spectral Energy')
ax1.set_title('Spectral Collapse in Model Merging', fontsize=12, fontweight='bold')
ax1.set_xlabel('Spatial Frequency', fontsize=10)
ax1.set_ylabel('Spectral Magnitude', fontsize=10)
ax1.legend(frameon=True, fontsize=9)
ax1.set_yscale('log')

# Plot 2: Spectral Alignment (FDSA restoration)
ax2.plot(freqs, collapsed_magnitude, 'r--', label='Merged Model (Collapsed)', linewidth=2)
ax2.plot(freqs, aligned_magnitude, 'b-', label='FDSA Calibrated (Ours)', linewidth=2.5)
ax2.plot(freqs, original_magnitude, 'g:', label='Target Profile', linewidth=1.5)
ax2.fill_between(freqs, collapsed_magnitude, aligned_magnitude, color='blue', alpha=0.15, label='Restored Frequencies')
ax2.set_title('FDSA Spectral Realignment', fontsize=12, fontweight='bold')
ax2.set_xlabel('Spatial Frequency', fontsize=10)
ax2.set_ylabel('Spectral Magnitude', fontsize=10)
ax2.legend(frameon=True, fontsize=9)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('spectral_concept.png', bbox_inches='tight')
print("Successfully generated spectral_concept.png!")
