import torch
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 5.5))

# Load Fisher sensitivities
layer_fisher = torch.load("checkpoints/layer_fisher.pt", map_location="cpu")

# Filter to keep the key weight parameter names
layers = []
sensitivities = []
for name, val in layer_fisher.items():
    if "weight" in name:
        # Clean name for presentation
        clean_name = name.replace(".weight", "")
        layers.append(clean_name)
        sensitivities.append(val)

# Convert to numpy
sensitivities = np.array(sensitivities)

# Ensure sorting or plotting order matches forward order (or alphabetical if standard)
# Plot
bars = ax.bar(layers, sensitivities, color='#2c3e50', edgecolor='black', alpha=0.85, width=0.6)
ax.set_yscale('log')
ax.set_ylabel('Average Parameter Fisher Information (FIM)', fontsize=12, fontweight='bold')
ax.set_xlabel('ResNet-18 Layer Name', fontsize=12, fontweight='bold')
ax.set_title('Layer-wise Empirical Fisher Information Prior', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.savefig("fisher_dist.pdf", dpi=300)
plt.savefig("fisher_dist.png", dpi=300)
print("Saved Fisher distribution plots to fisher_dist.pdf and fisher_dist.png")
