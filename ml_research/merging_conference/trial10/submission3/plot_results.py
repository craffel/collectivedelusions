import json
import matplotlib.pyplot as plt
import numpy as np

def plot():
    with open("experiment_results.json", "r") as f:
        data = json.load(f)
        
    trajectories = data["trajectories"]
    methods = list(trajectories.keys())
    
    plt.figure(figsize=(10, 5), dpi=300)
    
    # Define nice colors and line styles
    colors = {
        "Method A": "#94a3b8", # slate
        "Method B": "#f59e0b", # amber
        "Method C": "#d97706", # dark amber
        "Method D": "#ef4444", # red
        "Method E": "#06b6d4", # cyan
        "Method F": "#10b981", # emerald (ours)
        "Method G": "#6366f1", # indigo (ours)
        "Method H": "#ec4899",  # pink (ablation)
        "Method I": "#a855f7",  # purple (ours)
        "Method J": "#14b8a6",  # teal (ours)
        "Method K": "#3b82f6"   # blue (ours)
    }
    
    names = {
        "Method A": "Method A: Fixed TTA",
        "Method B": "Method B: CL W-Fisher + L2",
        "Method C": "Method C: CL W-Fisher + Angular",
        "Method D": "Method D: CP-AM",
        "Method E": "Method E: BK-AHR (Baseline SOTA)",
        "Method F": "Method F: BK-FWSAM (Ours Weight)",
        "Method G": "Method G: BK-DPS-SAM (Ours DPS)",
        "Method H": "Method H: Isotropic SAM-TTMM",
        "Method I": "Method I: BK-FWSAM + ACR-UF (Ours)",
        "Method J": "Method J: BK-MSAM (Ours Momentum)",
        "Method K": "Method K: BK-MSAM + SMR (Ours)"
    }
    
    for m in methods:
        accs = trajectories[m]
        # Smooth with a rolling average of 3 to make it readable
        smoothed = np.convolve(accs, np.ones(3)/3, mode='valid')
        # Pad to keep original length
        smoothed = np.pad(smoothed, (1, 1), mode='edge')
        plt.plot(smoothed, label=names[m], color=colors[m], linewidth=2 if m in ["Method F", "Method G", "Method H", "Method I", "Method J", "Method K"] else 1.5, alpha=1.0 if m in ["Method E", "Method F", "Method G", "Method H", "Method I", "Method J", "Method K"] else 0.7)
        
    # Mark the phase boundaries
    # Phase 0: 0-9, Phase 1: 10-19, Phase 2: 20-29, Phase 3: 30-39, Phase 4: 40-49
    phases = [
        (0, 9, "Clean\nMNIST", "#f1f5f9"),
        (10, 19, "Noisy\nMNIST", "#fee2e2"),
        (20, 29, "Clean\nFashion", "#f1f5f9"),
        (30, 39, "Noisy\nFashion", "#fee2e2"),
        (40, 49, "Novel\nKMNIST", "#ecfdf5")
    ]
    
    for start, end, name, col in phases:
        plt.axvspan(start, end, alpha=0.15, color=col)
        plt.text((start + end) / 2, 103, name, ha='center', fontsize=9, fontweight='bold')
        if end < 49:
            plt.axvline(end + 0.5, color='#94a3b8', linestyle='--', alpha=0.5)
            
    plt.title("Online Batch-by-Batch Accuracy Trajectory on Non-Stationary Test Stream", fontsize=12, fontweight='bold', pad=20)
    plt.xlabel("Streaming Test Batch Index", fontsize=10)
    plt.ylabel("Classification Accuracy (%)", fontsize=10)
    plt.ylim(-5, 115)
    plt.xlim(0, 49)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    
    plt.savefig("accuracy_trajectory.pdf", bbox_inches='tight')
    plt.savefig("accuracy_trajectory.png", bbox_inches='tight')
    print("Successfully plotted and saved accuracy_trajectory.pdf and accuracy_trajectory.png")

if __name__ == "__main__":
    plot()
