import numpy as np
import matplotlib.pyplot as plt

# Set style for academic publishing
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# ----------------------------------------------------
# Left Subplot: Continuous Soft-Trimming
# ----------------------------------------------------
x = np.linspace(0.0, 2.0, 500)
# TIES hard trimming: let's assume a threshold of 0.5 relative magnitude
y_ties = np.zeros_like(x)
y_ties[x >= 0.5] = x[x >= 0.5]

ax1.plot(x, x, label=r'Task Arithmetic ($\beta=0$)', color='#7f8c8d', linestyle='--', linewidth=1.5)
ax1.plot(x, y_ties, label=r'Hard Trimming (TIES, $p=0.2$)', color='#e74c3c', linestyle='-', linewidth=2)
ax1.plot(x, x * (x**0.5), label=r'ST-SCS ($\beta=0.5$)', color='#3498db', linestyle='-', linewidth=2)
ax1.plot(x, x * (x**1.0), label=r'ST-SCS ($\beta=1.0$, Ours)', color='#2ecc71', linestyle='-', linewidth=2.5)

ax1.set_title("Continuous Soft-Trimming", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel(r"Relative Input Magnitude $|\tau_{k, i}| / M_k^l$", fontsize=11)
ax1.set_ylabel(r"Effective Update $|\tilde{\tau}_{k, i}| / M_k^l$", fontsize=11)
ax1.set_xlim(0.0, 2.0)
ax1.set_ylim(0.0, 2.5)
ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
ax1.grid(True, linestyle=':', alpha=0.6)

# ----------------------------------------------------
# Right Subplot: Continuous Consensus Scaling
# ----------------------------------------------------
c = np.linspace(0.0, 1.0, 500)
ax2.plot(c, np.ones_like(c), label=r'Task Arithmetic ($\gamma=0$)', color='#7f8c8d', linestyle='--', linewidth=1.5)
ax2.plot(c, c**0.1, label=r'ST-SCS ($\gamma=0.1$)', color='#e67e22', linestyle='-', linewidth=2)
ax2.plot(c, c**0.5, label=r'ST-SCS ($\gamma=0.5$)', color='#9b59b6', linestyle='-', linewidth=2)
ax2.plot(c, c**1.0, label=r'ST-SCS ($\gamma=1.0$, Ours)', color='#2ecc71', linestyle='-', linewidth=2.5)
ax2.plot(c, c**2.0, label=r'ST-SCS ($\gamma=2.0$)', color='#34495e', linestyle='-', linewidth=2)

ax2.set_title("Continuous Consensus Scaling", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel(r"Continuous Consensus Ratio $c_i$", fontsize=11)
ax2.set_ylabel(r"Consensus Scaling Factor $c_i^\gamma$", fontsize=11)
ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(0.0, 1.05)
ax2.legend(loc="lower right", frameon=True, fontsize=9.5)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("st_scs_viz.pdf", format="pdf", bbox_inches='tight')
plt.savefig("st_scs_viz.png", format="png", dpi=300, bbox_inches='tight')
print("Successfully saved st_scs_viz.pdf and st_scs_viz.png")
