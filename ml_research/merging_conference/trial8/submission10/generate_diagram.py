import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Define style helpers
    box_style = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.5)
    title_font = dict(fontsize=10, fontweight='bold', family='sans-serif')
    text_font = dict(fontsize=9, family='sans-serif')
    arrow_style = dict(arrowstyle="->", lw=1.5, color="black")
    double_arrow_style = dict(arrowstyle="<->", lw=1.5, color="black")
    
    # 1. Input Box
    ax.text(1, 6, "Input Batch\n" + r"$\mathcal{X}^{(t)}$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#d1e7dd", ec="#0f5132", lw=1.5), **title_font)
    
    # Arrow to Feature Extraction
    ax.annotate("", xy=(2.2, 6), xytext=(1.6, 6), arrowprops=arrow_style)
    
    # 2. Backbone Module
    ax.text(3.1, 6, "Shared Backbone\n" + r"$f_\theta(x)$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#cfe2ff", ec="#084298", lw=1.5), **title_font)
    
    # Arrow from Backbone to Feature representation
    ax.annotate("", xy=(4.2, 6), xytext=(4.0, 6), arrowprops=arrow_style)
    
    # Features label
    ax.text(4.4, 6.2, "Features\n" + r"$F^{(t)}$", ha='center', va='center', **text_font)
    
    # Arrow split from Features to Distance and Expert Heads
    ax.annotate("", xy=(5.2, 5.0), xytext=(4.4, 6.0), arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="angle,angleA=0,angleB=-90,rad=5"))
    ax.annotate("", xy=(5.2, 6.0), xytext=(4.8, 6.0), arrowprops=arrow_style)
    
    # 3. Distance Computation
    ax.text(6.4, 5.0, "Prototype Distance\n" + r"$\tilde{d}_k^{(t)}$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#664d03", lw=1.5), **title_font)
    
    # 4. Expert Heads
    ax.text(6.4, 6.0, "Expert Heads\n" + r"$\{h_k\}$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#f8d7da", ec="#842029", lw=1.5), **title_font)
    
    # Arrow from Expert Heads to Routing & Gating
    ax.annotate("", xy=(7.6, 6.0), xytext=(8.2, 6.0), arrowprops=dict(arrowstyle="<-", lw=1.5))
    # Arrow from Distance to Routing
    ax.annotate("", xy=(7.6, 5.0), xytext=(8.0, 5.0), arrowprops=dict(arrowstyle="<-", lw=1.5))
    
    # 5. Routing Module Box (dashed boundary to enclose routing logic)
    routing_rect = patches.FancyBboxPatch((7.6, 4.4), 1.9, 2.1, boxstyle="round,pad=0.1", fc="#f8f9fa", ec="#6c757d", ls="--", lw=1.2)
    ax.add_patch(routing_rect)
    ax.text(8.55, 6.2, "Routing Module", ha='center', va='center', fontweight='bold', fontsize=9, color="#495057", family='sans-serif')
    ax.text(8.55, 5.3, "Routing Weights\n" + r"$w_k^{(t)} = \mathrm{Softmax}(\mathbf{l}^{(t)})$" + "\n" + r"$\mathbf{l}^{(t)} = -\tilde{\mathbf{d}}^{(t)}/\tau + cw \cdot \mathbf{C}^{(t)}$", ha='center', va='center', **text_font)
    
    # Arrow down to Ensembled Prediction
    ax.annotate("", xy=(8.55, 3.8), xytext=(8.55, 4.4), arrowprops=arrow_style)
    
    # 6. Ensembled Prediction
    ax.text(8.55, 3.4, "Ensembled Prediction\n" + r"$P(y|x) = \sum_k w_k^{(t)} P_k(y|x)$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#e2d9f3", ec="#563d7c", lw=1.5), **title_font)
    
    # Arrow from Ensembled Prediction to Gating
    ax.annotate("", xy=(6.5, 3.4), xytext=(7.1, 3.4), arrowprops=arrow_style)
    
    # 7. Dual-Tier Gating Box (large grey box enclosing the gating logic)
    gating_rect = patches.FancyBboxPatch((1.0, 0.8), 5.0, 3.2, boxstyle="round,pad=0.1", fc="#f1f3f5", ec="#adb5bd", lw=1.5)
    ax.add_patch(gating_rect)
    ax.text(3.5, 3.8, "DUAL-TIER GATING MECHANISM", ha='center', va='center', fontweight='bold', fontsize=10, color="#212529", family='sans-serif')
    
    # Gate A: OOD Gating
    ax.text(2.2, 2.8, "Gate A: OOD Gating\n" + r"$\mathrm{OOD}^{(t)} = \mathrm{True}$ if:" + "\n" + r"$H(P) > \tau_H$ or $\min_k \tilde{d}_k^{(t)} > \tau_d$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#e2e3e5", ec="#383d41", lw=1.2), **text_font)
    
    # Gate B: Confidence-Calibrated Gating
    ax.text(4.8, 2.8, "Gate B: Confidence-Calibrated\n" + r"$\mathrm{Mismatch}^{(t)} = \mathrm{True}$ if:" + "\n" + r"$C_{k^*}^{(t)} < \max_{j \neq k^*} C_j^{(t)}$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#e2e3e5", ec="#383d41", lw=1.2), **text_font)
    
    # Arrow from Distance to OOD Gating
    ax.annotate("", xy=(2.2, 3.1), xytext=(6.4, 4.6), arrowprops=dict(arrowstyle="->", lw=1.2, color="#495057", connectionstyle="angle,angleA=90,angleB=180,rad=3"))
    # Arrow from Expert Heads to Gate B
    ax.annotate("", xy=(4.8, 3.1), xytext=(6.4, 5.6), arrowprops=dict(arrowstyle="->", lw=1.2, color="#495057", connectionstyle="angle,angleA=90,angleB=180,rad=3"))
    
    # Gate Decision Box
    ax.text(3.5, 1.6, "Are both gates False?\n" + r"$\neg \mathrm{OOD}^{(t)} \wedge \neg \mathrm{Mismatch}^{(t)}$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#664d03", lw=1.2), **title_font)
    ax.annotate("", xy=(3.5, 1.9), xytext=(2.2, 2.4), arrowprops=arrow_style)
    ax.annotate("", xy=(3.5, 1.9), xytext=(4.8, 2.4), arrowprops=arrow_style)
    
    # Yes / No outputs
    ax.annotate("", xy=(3.5, 0.4), xytext=(3.5, 1.2), arrowprops=arrow_style)
    ax.text(3.7, 0.7, "YES", fontsize=9, fontweight='bold', color='green', family='sans-serif')
    
    ax.annotate("", xy=(0.3, 1.6), xytext=(2.2, 1.6), arrowprops=arrow_style)
    ax.text(1.2, 1.8, "NO", fontsize=9, fontweight='bold', color='red', family='sans-serif')
    
    # 8. Update Action Box
    ax.text(3.5, -0.3, "Update Prototype & Calibration\n" + r"$\mu_{k^*}^{(t)} = (1-\alpha)\mu_{k^*}^{(t-1)} + \alpha \bar{F}^{(t)}$" + "\n" + r"$d_{\mathrm{cal}, k^*}^{(t)} = (1-\alpha)d_{\mathrm{cal}, k^*}^{(t-1)} + \alpha d_{k^*}^{(t)}$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#d1e7dd", ec="#0f5132", lw=1.5), **title_font)
    
    # 9. Gate Update Box
    ax.text(-0.8, 1.6, "Gate Update (No Change)\n" + r"$\mu_k^{(t)} = \mu_k^{(t-1)}$" + "\n" + r"$d_{\mathrm{cal}, k}^{(t)} = d_{\mathrm{cal}, k}^{(t-1)}$", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#f8d7da", ec="#842029", lw=1.5), **title_font)
    
    # Draw arrow from Features to Update box
    ax.annotate("", xy=(3.5, 0.4), xytext=(4.4, 5.8), arrowprops=dict(arrowstyle="->", lw=1.2, color="#495057", connectionstyle="angle,angleA=180,angleB=90,rad=5"))
    
    plt.tight_layout()
    plt.savefig("gated_ema_proto_arch.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print("Beautiful system architecture diagram generated and saved to gated_ema_proto_arch.pdf!")

if __name__ == "__main__":
    draw_architecture()
