import matplotlib.pyplot as plt
import numpy as np

# Data from our calculations
# For SimpleCNN:
# Layer: conv1.weight (1), conv2.weight (2), fc.weight (3)
cnn_layers = ['conv1.weight', 'conv2.weight', 'fc.weight']
cnn_alphas = [0.5559, 0.5780, 0.5780]
cnn_lambdas = [1.8005, 1.7304, 1.7301]

# For CLIP ViT-B/32 Blocks (0 to 11)
# We will average the three projections (Attn Out Proj, MLP c_fc, MLP c_proj) per block
clip_blocks = list(range(12))
clip_alphas = [
    (0.5772 + 0.5774 + 0.5772) / 3.0, # Block 0
    (0.5778 + 0.5774 + 0.5775) / 3.0, # Block 1
    (0.5768 + 0.5772 + 0.5774) / 3.0, # Block 2
    (0.5774 + 0.5771 + 0.5773) / 3.0, # Block 3
    (0.5777 + 0.5773 + 0.5771) / 3.0, # Block 4
    (0.5770 + 0.5774 + 0.5771) / 3.0, # Block 5
    (0.5776 + 0.5774 + 0.5773) / 3.0, # Block 6
    (0.5773 + 0.5774 + 0.5773) / 3.0, # Block 7
    (0.5778 + 0.5774 + 0.5773) / 3.0, # Block 8
    (0.5770 + 0.5775 + 0.5773) / 3.0, # Block 9
    (0.5774 + 0.5773 + 0.5774) / 3.0, # Block 10
    (0.5774 + 0.5774 + 0.5774) / 3.0, # Block 11
]
clip_lambdas = [1.0 / a for a in clip_alphas]

# Let's set up a beautiful publication-quality style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

# Left Panel: SimpleCNN
x_cnn = np.arange(len(cnn_layers))
ax1.plot(x_cnn, cnn_alphas, marker='o', color='#1f77b4', linewidth=2, label=r'Alignment Ratio $\alpha^l$')
ax1.set_ylabel(r'Alignment Ratio $\alpha^l$', color='#1f77b4', fontsize=11)
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_ylim(0.50, 0.65)

ax1_right = ax1.twinx()
ax1_right.plot(x_cnn, cnn_lambdas, marker='s', color='#ff7f0e', linewidth=2, linestyle='--', label=r'Scaling Factor $\lambda^l$')
ax1_right.set_ylabel(r'Scaling Factor $\lambda^l = 1/\alpha^l$', color='#ff7f0e', fontsize=11)
ax1_right.tick_params(axis='y', labelcolor='#ff7f0e')
ax1_right.set_ylim(1.50, 2.00)

ax1.set_xticks(x_cnn)
ax1.set_xticklabels(['conv1', 'conv2', 'fc'], fontsize=10)
ax1.set_title('(a) SimpleCNN (Grayscale)', fontsize=12, fontweight='bold', pad=10)
ax1.grid(True, linestyle=':', alpha=0.6)

# Horizontal reference line for K=3 orthogonal limit (alpha = 1/sqrt(3) ≈ 0.5774, lambda = sqrt(3) ≈ 1.732)
ax1.axhline(1.0/np.sqrt(3.0), color='gray', linestyle=':', alpha=0.7, label='Orthogonal Limit')

# Right Panel: CLIP ViT-B/32 Blocks
ax2.plot(clip_blocks, clip_alphas, marker='o', color='#1f77b4', linewidth=2, label=r'Alignment Ratio $\alpha^l$')
ax2.set_ylabel(r'Alignment Ratio $\alpha^l$', color='#1f77b4', fontsize=11)
ax2.tick_params(axis='y', labelcolor='#1f77b4')
ax2.set_ylim(0.57, 0.585)

ax2_right = ax2.twinx()
ax2_right.plot(clip_blocks, clip_lambdas, marker='s', color='#ff7f0e', linewidth=2, linestyle='--', label=r'Scaling Factor $\lambda^l$')
ax2_right.set_ylabel(r'Scaling Factor $\lambda^l = 1/\alpha^l$', color='#ff7f0e', fontsize=11)
ax2_right.tick_params(axis='y', labelcolor='#ff7f0e')
ax2_right.set_ylim(1.71, 1.75)

ax2.set_xlabel('Transformer Block Index', fontsize=11)
ax2.set_xticks(clip_blocks)
ax2.set_title('(b) CLIP ViT-B/32 Visual Encoder', fontsize=12, fontweight='bold', pad=10)
ax2.grid(True, linestyle=':', alpha=0.6)

ax2.axhline(1.0/np.sqrt(3.0), color='gray', linestyle=':', alpha=0.7)

plt.tight_layout()

# Save plot to both directories
plt.savefig('results/fig_layerwise.png', bbox_inches='tight')
import os
os.makedirs('submission/results', exist_ok=True)
plt.savefig('submission/results/fig_layerwise.png', bbox_inches='tight')
print("Successfully generated and saved layer-wise stats plot to results/fig_layerwise.png and submission/results/fig_layerwise.png")
