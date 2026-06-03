import matplotlib.pyplot as plt
import numpy as np
import os

def generate_plots():
    results_path = "robustness_results.txt"
    if not os.path.exists(results_path):
        print(f"File {results_path} not found. Cannot plot robustness yet.")
        return
        
    # Read the data
    data = []
    with open(results_path, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split(",")
        for line in lines[1:]:
            parts = line.strip().split(",")
            data.append({
                "merge": parts[0],
                "calib": parts[1],
                "pert": parts[2],
                "intensity": float(parts[3]),
                "avg_acc": float(parts[4])
            })
            
    # Professional plot styles
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'grid.alpha': 0.6
    })
    
    # We will create a 1x2 side-by-side plot
    # Left: Noise Robustness (under TA lambda=0.5)
    # Right: Blur Robustness (under TA lambda=0.5)
    # We will show WA and TA on both or just focus on TA since it's our main paradigm
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    
    # Filter for TA lambda=0.5
    ta_data = [d for d in data if d["merge"] == "TA"]
    
    # Define color palette & line styles
    colors = {
        "No Calibration": "#e74c3c", # Red for uncalibrated
        "Real Joint Multi-Task Calib": "#f1c40f", # Yellow for real joint
        "Generative BN-Matching Calib (Ours)": "#2c3e50" # Dark Blue for ours
    }
    
    markers = {
        "No Calibration": "o",
        "Real Joint Multi-Task Calib": "s",
        "Generative BN-Matching Calib (Ours)": "D"
    }
    
    line_styles = {
        "No Calibration": ":",
        "Real Joint Multi-Task Calib": "-.",
        "Generative BN-Matching Calib (Ours)": "-"
    }

    labels = {
        "No Calibration": "No Calibration",
        "Real Joint Multi-Task Calib": "Real Joint Calib (Mixture)",
        "Generative BN-Matching Calib (Ours)": "DF-Calib-Gen (Ours)"
    }
    
    # 1. Plot Noise Robustness
    noise_data = [d for d in ta_data if d["pert"] == "noise"]
    calibs = list(set([d["calib"] for d in noise_data]))
    
    for calib in ["No Calibration", "Real Joint Multi-Task Calib", "Generative BN-Matching Calib (Ours)"]:
        calib_subset = [d for d in noise_data if d["calib"] == calib]
        # Sort by intensity
        calib_subset = sorted(calib_subset, key=lambda x: x["intensity"])
        intensities = [x["intensity"] for x in calib_subset]
        accuracies = [x["avg_acc"] for x in calib_subset]
        
        ax1.plot(
            intensities, accuracies, 
            color=colors[calib], 
            marker=markers[calib], 
            linestyle=line_styles[calib], 
            linewidth=2 if calib == "Generative BN-Matching Calib (Ours)" else 1.5,
            label=labels[calib]
        )
        
    ax1.set_xlabel("Additive Gaussian Noise ($\sigma$)", fontweight="bold")
    ax1.set_ylabel("Average Multi-Task Accuracy (%)", fontweight="bold")
    ax1.set_title("Robustness to Additive Noise", fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(-0.02, 0.32)
    ax1.set_xticks([0.0, 0.1, 0.2, 0.3])
    
    # 2. Plot Blur Robustness
    blur_data = [d for d in ta_data if d["pert"] == "blur"]
    # Include the 0.0 noise data as 0.0 blur
    zero_blur_data = [d for d in ta_data if d["pert"] == "noise" and d["intensity"] == 0.0]
    for d in zero_blur_data:
        copied = d.copy()
        copied["pert"] = "blur"
        copied["intensity"] = 0.0
        blur_data.append(copied)
        
    for calib in ["No Calibration", "Real Joint Multi-Task Calib", "Generative BN-Matching Calib (Ours)"]:
        calib_subset = [d for d in blur_data if d["calib"] == calib]
        calib_subset = sorted(calib_subset, key=lambda x: x["intensity"])
        intensities = [x["intensity"] for x in calib_subset]
        accuracies = [x["avg_acc"] for x in calib_subset]
        
        ax2.plot(
            intensities, accuracies, 
            color=colors[calib], 
            marker=markers[calib], 
            linestyle=line_styles[calib], 
            linewidth=2 if calib == "Generative BN-Matching Calib (Ours)" else 1.5,
            label=labels[calib]
        )
        
    ax2.set_xlabel("Gaussian Blur Sigma ($\sigma_{\\text{blur}}$)", fontweight="bold")
    ax2.set_ylabel("Average Multi-Task Accuracy (%)", fontweight="bold")
    ax2.set_title("Robustness to Image Blur", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_xlim(-0.1, 2.1)
    ax2.set_xticks([0.0, 1.0, 1.5, 2.0])
    
    # Single legend for the whole figure
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig("robustness_results.pdf", bbox_inches="tight")
    plt.savefig("robustness_results.png", bbox_inches="tight")
    print("Robustness plots generated successfully!")

if __name__ == "__main__":
    generate_plots()
