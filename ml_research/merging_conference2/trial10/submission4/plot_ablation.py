import os
import json
import matplotlib.pyplot as plt

def main():
    print("Generating BN calibration sample size ablation plot...")
    with open("ablation_bn_results_cpu.json", "r") as f:
        data = json.load(f)
        
    # Group results by method
    method_data = {}
    for run in data:
        m = run["method"]
        # Clean method names for plotting
        if "QCOT" in m:
            m = "QCOT"
        elif "QWC" in m:
            m = "QWC"
        elif "CWSS-QC" in m:
            m = "CWSS-QC (Ours)"
        elif "CWSS" in m:
            m = "CWSS (Ours)"
            
        if m not in method_data:
            method_data[m] = {"x": [], "y": []}
        method_data[m]["x"].append(run["bn_calib"])
        method_data[m]["y"].append(run["cifar10_acc"])
        
    # Sort points by x (sample size) for plotting lines
    for m in method_data:
        sorted_pts = sorted(zip(method_data[m]["x"], method_data[m]["y"]))
        method_data[m]["x"] = [x for x, y in sorted_pts]
        method_data[m]["y"] = [y for x, y in sorted_pts]
        
    # Create plot
    plt.figure(figsize=(7, 5.2), dpi=300)
    
    # Custom styling
    colors = {
        "WA": "#7f7f7f",
        "TA": "#bcbd22",
        "QWC": "#17becf",
        "QCOT": "#e377c2",
        "CWSS (Ours)": "#1f77b4",
        "CWSS-QC (Ours)": "#2ca02c"
    }
    
    markers = {
        "WA": "o",
        "TA": "s",
        "QWC": "^",
        "QCOT": "d",
        "CWSS (Ours)": "D",
        "CWSS-QC (Ours)": "v"
    }
    
    linestyles = {
        "WA": "--",
        "TA": "--",
        "QWC": "--",
        "QCOT": "-",
        "CWSS (Ours)": "-",
        "CWSS-QC (Ours)": "-"
    }
    
    for m in ["WA", "TA", "QCOT", "QWC", "CWSS (Ours)", "CWSS-QC (Ours)"]:
        if m in method_data:
            plt.plot(
                method_data[m]["x"], 
                method_data[m]["y"], 
                label=m,
                color=colors[m],
                marker=markers[m],
                linestyle=linestyles[m],
                linewidth=2,
                markersize=6
            )
            
    plt.xlabel("Number of Calibration Samples ($N$)", fontsize=11, fontweight="bold")
    plt.ylabel("CIFAR-10 Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Calibration Data Efficiency under INT4 Quantization", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    
    # Save figure
    os.makedirs("results", exist_ok=True)
    out_path = "results/bn_sample_ablation.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Ablation plot saved successfully to {out_path}!")

if __name__ == "__main__":
    main()
