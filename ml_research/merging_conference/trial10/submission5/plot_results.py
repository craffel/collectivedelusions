import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the results
    data = np.load("ttmm_results.npy", allow_pickle=True).item()
    
    # Let's inspect the keys and confirm accuracy
    for regime, metrics in data.items():
        print(f"Regime: {regime}")
        print(f"  Overall Acc: {metrics['overall_acc']*100:.2f}%")
        print(f"  Phase Accs: {[round(x*100, 2) for x in metrics['phase_accs']]}")
        print(f"  Curv len: {len(metrics['curv'])}")
        print(f"  LR len: {len(metrics['lrs'])}")
        print(f"  Damp len: {len(metrics['damps'])}")
        
    # Let's set up a beautiful plot
    # Two panels: 
    # Left: Curvature (loss gap) across phases
    # Right: Adaptive learning rate and damping
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Phase boundaries
    # Each phase is 10 batches. Boundaries are at 10, 20, 30, 40.
    boundaries = [10, 20, 30, 40]
    phase_labels = [
        "Clean\nMNIST", 
        "Noisy\nMNIST", 
        "Clean\nFashion", 
        "Noisy\nFashion", 
        "Novel\nKMNIST"
    ]
    phase_colors = ["#f9f9f9", "#eaeaea", "#f9f9f9", "#eaeaea", "#ffebeb"]
    
    # ------------------ PANEL 1: Curvature ------------------
    ax1 = axes[0]
    batches = np.arange(50)
    
    # Get curvature from cg_mttmm (standard)
    curv_std = np.array(data["cg_mttmm"]["curv"])
    curv_tuned = np.array(data["cg_mttmm_tuned"]["curv"])
    
    # Shade phases
    for i in range(5):
        start = i * 10
        end = (i + 1) * 10
        ax1.axvspan(start - 0.5, end - 0.5, color=phase_colors[i], alpha=0.5, zorder=0)
        # Place label at the top
        ax1.text(start + 4.5, max(curv_std.max(), curv_tuned.max()) * 1.1, phase_labels[i], 
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
        
    ax1.plot(batches, curv_std, marker="o", color="#1f77b4", linewidth=2.0, label=r"Standard ($\rho=0.05$)")
    ax1.plot(batches, curv_tuned, marker="s", color="#ff7f0e", linewidth=2.0, linestyle="--", label=r"Tuned ($\rho=0.03$)")
    
    ax1.set_xlabel("Batch Index in Streaming Evaluation", fontsize=11)
    ax1.set_ylabel(r"Weight Perturbation Loss Gap $\Delta \mathcal{L}$ (Curvature)", fontsize=11)
    ax1.set_title("On-the-Fly Loss Landscape Curvature Estimation", fontsize=12, fontweight="bold", pad=25)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(-0.5, 49.5)
    ax1.set_ylim(0, max(curv_std.max(), curv_tuned.max()) * 1.2)
    ax1.legend(loc="upper left")
    
    # ------------------ PANEL 2: Adaptive Parameters (Tuned) ------------------
    ax2 = axes[1]
    lrs_tuned = np.array(data["cg_mttmm_tuned"]["lrs"])
    damps_tuned = np.array(data["cg_mttmm_tuned"]["damps"])
    
    # Shade phases
    for i in range(5):
        start = i * 10
        end = (i + 1) * 10
        ax2.axvspan(start - 0.5, end - 0.5, color=phase_colors[i], alpha=0.5, zorder=0)
        # Place label at the top
        ax2.text(start + 4.5, 170, phase_labels[i], 
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
        
    color_lr = "#2ca02c"
    color_damp = "#d62728"
    
    # Left axis for Learning Rate
    ax2_lr = ax2
    line_lr, = ax2_lr.plot(batches, lrs_tuned, marker="^", color=color_lr, linewidth=2.0, label="Adaptive LR (Left)")
    ax2_lr.set_xlabel("Batch Index in Streaming Evaluation", fontsize=11)
    ax2_lr.set_ylabel(r"Adaptive Learning Rate $\eta_{adaptive}$", color=color_lr, fontsize=11)
    ax2_lr.tick_params(axis="y", labelcolor=color_lr)
    ax2_lr.set_ylim(0, 180)
    
    # Right axis for Damping
    ax2_damp = ax2.twinx()
    line_damp, = ax2_damp.plot(batches, damps_tuned, marker="v", color=color_damp, linewidth=2.0, linestyle="-.", label="Adaptive Damping (Right)")
    ax2_damp.set_ylabel(r"Adaptive Damping parameter $\epsilon_{adaptive}$", color=color_damp, fontsize=11)
    ax2_damp.tick_params(axis="y", labelcolor=color_damp)
    ax2_damp.set_ylim(0, damps_tuned.max() * 1.2)
    
    ax2.set_title("Curvature-Guided Adaptive Optimization Trajectory", fontsize=12, fontweight="bold", pad=25)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlim(-0.5, 49.5)
    
    # Combined legend
    lines = [line_lr, line_damp]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")
    
    plt.tight_layout()
    plt.savefig("curvature_and_adaptation.pdf", bbox_inches="tight")
    print("Beautiful plot saved to curvature_and_adaptation.pdf")

if __name__ == "__main__":
    main()
