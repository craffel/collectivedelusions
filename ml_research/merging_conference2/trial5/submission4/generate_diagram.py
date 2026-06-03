import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'text.usetex': False
})

fig, ax = plt.subplots(figsize=(10.5, 3.2))

# Hide axes
ax.axis('off')
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 3.2)

# Helper function to draw a box
def draw_box(ax, x, y, w, h, text, title="", color='#eef4f8', border_color='#3182bd'):
    box = patches.FancyBboxPatch((x + 0.05, y + 0.05), w - 0.1, h - 0.1, 
                                 boxstyle="round,pad=0.03", 
                                 linewidth=1.5, edgecolor=border_color, facecolor=color)
    ax.add_patch(box)
    if title:
        ax.text(x + w/2, y + h - 0.25, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1c1c1c')
        ax.text(x + w/2, y + (h - 0.2)/2, text, ha='center', va='center', fontsize=9, color='#333333')
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9.5, color='#1c1c1c')

# Draw the boxes
# Box 1: Multi-task Model Merging
draw_box(ax, 0.1, 0.8, 2.2, 1.5, 
         "Weight Averaging or\nTask Arithmetic on\nFine-tuned Expert Models", 
         "1. Model Merging", color='#f7fbff', border_color='#08519c')

# Box 2: Fourier Spectral Analysis
draw_box(ax, 2.7, 0.8, 2.2, 1.5, 
         "Run $N$ calibration samples;\nCompute 2D FFT magnitudes\nto extract spectral map $\\Gamma^*$", 
         "2. Spectral Mapping", color='#f7fcf5', border_color='#006d2c')

# Box 3: Spatial Projection & Truncation
draw_box(ax, 5.3, 0.8, 2.2, 1.5, 
         "Map $\\Gamma^*$ to spatial domain\nvia IFFT; truncate to local\n$3 \\times 3$ depthwise filter $W_{\\text{dw}}$", 
         "3. Spatial Truncation", color='#fcfbfd', border_color='#54278f')

# Box 4: BatchNorm Folding & Mathematical Fusion
draw_box(ax, 7.9, 0.8, 2.5, 1.5, 
         "Fold BN into preceding Conv;\nConvolve Conv weights with $W_{\\text{dw}}$:\n$W'' = W_{\\text{dw}} * W'$\n$b'' = b' \\cdot \\sum W_{\\text{dw}}$", 
         "4. Associative Fusion", color='#fff5f0', border_color='#a50f15')

# Draw arrows
def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color='#444444', lw=2, mutation_scale=15))

draw_arrow(ax, 2.35, 1.55, 2.65, 1.55)
draw_arrow(ax, 4.95, 1.55, 5.25, 1.55)
draw_arrow(ax, 7.55, 1.55, 7.85, 1.55)

# Add a caption/title inside the plot or let LaTeX handle it (LaTeX handles figure caption)
ax.text(1.2, 0.3, "Input: Merged Model", ha='center', va='center', fontsize=9.5, style='italic', color='#555555')
ax.text(3.8, 0.3, "Fourier Domain Mapping", ha='center', va='center', fontsize=9.5, style='italic', color='#555555')
ax.text(6.4, 0.3, "IFFT & Occam's Razor", ha='center', va='center', fontsize=9.5, style='italic', color='#555555')
ax.text(9.15, 0.3, "Output: Single Compiled Conv Layer\nwith Zero Runtime Overhead", ha='center', va='center', fontsize=9.5, style='italic', color='#555555')

plt.tight_layout()
plt.savefig("pipeline_diagram.pdf", bbox_inches='tight', dpi=300)
plt.close()
print("Saved pipeline_diagram.pdf")
