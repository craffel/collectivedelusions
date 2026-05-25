import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracies():
    if not os.path.exists("tta_results.json"):
        print("tta_results.json not found. Run run_tta.py first.")
        return
        
    with open("tta_results.json", "r") as f:
        results = json.load(f)
        
    environments = ["Clean", "Gaussian Noise", "Gaussian Blur", "Contrast"]
    methods = ["Static", "Static Fisher", "Standard TTA", "FiT-Merge (Ours)"]
    streams = ["Alternating", "Sequential"]
    
    # We will plot for both teacher-supervised and teacher-free modes
    for loss_mode in ["teacher-supervised", "teacher-free"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, stream_type in enumerate(streams):
            ax = axes[idx]
            
            # Gather accuracies
            data = {method: [] for method in methods}
            for env in environments:
                for method in methods:
                    acc = results[loss_mode][stream_type][env][method]["accuracy"]
                    data[method].append(acc)
                    
            x = np.arange(len(environments))
            width = 0.2
            
            ax.bar(x - 1.5 * width, data["Static"], width, label="Static Merged", color="#7f8c8d")
            ax.bar(x - 0.5 * width, data["Static Fisher"], width, label="Static Fisher", color="#9b59b6")
            ax.bar(x + 0.5 * width, data["Standard TTA"], width, label="Standard TTA", color="#3498db")
            ax.bar(x + 1.5 * width, data["FiT-Merge (Ours)"], width, label="FiT-Merge (Ours)", color="#e74c3c")
            
            ax.set_title(f"{stream_type} Stream", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(environments, fontsize=11)
            ax.set_ylabel("Multi-Task Accuracy (%)", fontsize=12)
            ax.set_ylim(0, 105)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            if idx == 0:
                ax.legend(fontsize=11, loc="upper right")
                
        plt.suptitle(f"Test-Time Adaptation Performance ({loss_mode.replace('-', ' ').title()})", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(f"tta_accuracies_{loss_mode}.png", dpi=300)
        plt.close()
        print(f"Saved tta_accuracies_{loss_mode}.png")

def plot_adaptation_curve():
    if not os.path.exists("tta_results.json"):
        print("tta_results.json not found. Run run_tta.py first.")
        return
        
    with open("tta_results.json", "r") as f:
        results = json.load(f)
        
    stream_type = "Sequential"
    env = "Gaussian Noise"
    methods = ["Static", "Static Fisher", "Standard TTA", "FiT-Merge (Ours)"]
    
    for loss_mode in ["teacher-supervised", "teacher-free"]:
        plt.figure(figsize=(12, 5))
        
        for method in methods:
            # For Static Fisher in teacher-free mode, it has the same accuracies as teacher-supervised, so we load from there
            src_loss_mode = "teacher-supervised" if method in ["Static", "Static Fisher"] else loss_mode
            batch_accs = results[src_loss_mode][stream_type][env][method]["batch_accuracies"]
            # Smooth with a rolling average of window size 5
            smoothed_accs = np.convolve(batch_accs, np.ones(5)/5, mode='valid')
            
            if method == "Static":
                color = "#7f8c8d"
            elif method == "Static Fisher":
                color = "#9b59b6"
            elif method == "Standard TTA":
                color = "#3498db"
            else:
                color = "#e74c3c"
                
            plt.plot(np.arange(len(smoothed_accs)) + 2, smoothed_accs, label=method, color=color, linewidth=2)
            
        # Draw vertical lines for task-transition boundaries
        # In sequential stream, there are 150 total batches of size 64: 50 batches per task
        # Since we smoothed with window size 5, the transition boundaries are slightly shifted but still clear around batch 50 and 100
        plt.axvline(x=50, color='gray', linestyle='--', alpha=0.8)
        plt.axvline(x=100, color='gray', linestyle='--', alpha=0.8)
        
        # Add labels for task regions
        plt.text(20, 95, "MNIST", fontsize=12, fontweight="bold", color="gray")
        plt.text(65, 95, "FashionMNIST", fontsize=12, fontweight="bold", color="gray")
        plt.text(115, 95, "KMNIST", fontsize=12, fontweight="bold", color="gray")
        
        plt.title(f"Sequential Test-Time Adaptation Dynamics ({env}, {loss_mode.replace('-', ' ').title()})", fontsize=14, fontweight="bold")
        plt.xlabel("Test-Stream Step (Batch)", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.xlim(0, 150)
        plt.ylim(0, 105)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11, loc="lower left")
        plt.tight_layout()
        plt.savefig(f"tta_dynamics_{loss_mode}.png", dpi=300)
        plt.close()
        print(f"Saved tta_dynamics_{loss_mode}.png")

if __name__ == "__main__":
    plot_accuracies()
    plot_adaptation_curve()
