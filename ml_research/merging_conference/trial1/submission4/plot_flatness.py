import json
import matplotlib.pyplot as plt

def generate_plot():
    # Read flatness results
    with open('flatness_results.json', 'r') as f:
        data = json.load(f)
    
    # Extract data
    noise_scales = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    
    # Map from json keys to labels and style
    methods_config = {
        "OrthoMerge (Standard)": {
            "label": "OrthoMerge (Standard)",
            "color": "#7F8C8D", # Grey
            "marker": "o",
            "linestyle": "--"
        },
        "D-SMM (rho=0.10)": {
            "label": r"D-SMM ($\rho = 0.10$)",
            "color": "#3498DB", # Blue
            "marker": "s",
            "linestyle": "-"
        },
        "D-SMM (rho=0.20)": {
            "label": r"D-SMM ($\rho = 0.20$)",
            "color": "#E74C3C", # Red
            "marker": "^",
            "linestyle": "-"
        }
    }
    
    plt.figure(figsize=(6, 4))
    
    for key, config in methods_config.items():
        if key in data:
            # Reconstruct accuracies for the exact noise_scales keys
            accuracies = [data[key][str(ns)] for ns in noise_scales]
            plt.plot(
                noise_scales, 
                accuracies, 
                label=config["label"],
                color=config["color"],
                marker=config["marker"],
                linestyle=config["linestyle"],
                linewidth=2,
                markersize=6
            )
            
    plt.xlabel(r"Noise Perturbation Scale ($\sigma$)", fontsize=11)
    plt.ylabel("Multi-Task Average Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    
    plt.savefig('flatness_curve.pdf', format='pdf', dpi=300)
    print("Flatness curve generated successfully as flatness_curve.pdf!")

if __name__ == '__main__':
    generate_plot()
