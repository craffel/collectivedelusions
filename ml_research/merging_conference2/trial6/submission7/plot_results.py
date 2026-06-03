import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    try:
        with open("experiment_results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("experiment_results.json not found yet. Cannot plot.")
        return

    # Extract data
    methods = ["WA", "WA_BN", "TA", "TA_BN"]
    colors = {
        "WA": "tab:blue",
        "WA_BN": "tab:cyan",
        "TA": "tab:orange",
        "TA_BN": "tab:red"
    }
    markers = {
        "WA": "o",
        "WA_BN": "x",
        "TA": "s",
        "TA_BN": "^"
    }
    
    # 1. Plot CKA vs K
    plt.figure(figsize=(8, 6))
    layers = ["layer1", "layer2", "layer3", "layer4"]
    layer_styles = {"layer1": "-", "layer2": "--", "layer3": "-.", "layer4": ":"}
    layer_markers = {"layer1": "o", "layer2": "s", "layer3": "^", "layer4": "D"}
    
    # We will plot CKA for WA and TA separately or together
    # Let's plot for WA to keep it clean, as CKA is very similar between WA and TA
    K_vals = [res["K"] for res in data["WA"]]
    for layer in layers:
        cka_vals = [res["avg_cka"][layer] for res in data["WA"]]
        plt.plot(K_vals, cka_vals, label=f"WA - {layer.capitalize()}", 
                 linestyle=layer_styles[layer], marker=layer_markers[layer], linewidth=2)
        
    plt.xlabel("Number of Merged Tasks (K)", fontsize=12)
    plt.ylabel("Average CKA with Experts", fontsize=12)
    plt.title("Early vs. Late Layer Representation Similarity (WA)", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(K_vals)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    plt.savefig("cka_vs_k.png", dpi=300)
    plt.close()
    print("Saved cka_vs_k.png")
    
    # 2. Plot Routing Accuracy vs K
    plt.figure(figsize=(8, 6))
    for method in methods:
        K_vals = [res["K"] for res in data[method]]
        routing_accs = [res["routing_acc"] for res in data[method]]
        plt.plot(K_vals, routing_accs, label=f"{method} - MSPR Routing", 
                 color=colors[method], marker=markers[method], linewidth=2)
        
    plt.xlabel("Number of Merged Tasks (K)", fontsize=12)
    plt.ylabel("Routing Accuracy (%)", fontsize=12)
    plt.title("MSPR Layer 2 Task Routing Accuracy vs. Task Count", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(K_vals)
    plt.ylim(0, 105)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("routing_vs_k.png", dpi=300)
    plt.close()
    print("Saved routing_vs_k.png")
    
    # 3. Plot Downstream Multi-Task Accuracy vs K (Oracle vs MSPR)
    plt.figure(figsize=(10, 6))
    for method in methods:
        K_vals = [res["K"] for res in data[method]]
        oracle_accs = [res["oracle_acc"] for res in data[method]]
        mspr_accs = [res["mspr_acc"] for res in data[method]]
        
        plt.plot(K_vals, oracle_accs, label=f"{method} - Oracle Gated", 
                 color=colors[method], linestyle="-", marker=markers[method], linewidth=2)
        plt.plot(K_vals, mspr_accs, label=f"{method} - MSPR Routed", 
                 color=colors[method], linestyle="--", marker=markers[method], linewidth=1.5, alpha=0.8)
        
    plt.xlabel("Number of Merged Tasks (K)", fontsize=12)
    plt.ylabel("Downstream Multi-Task Accuracy (%)", fontsize=12)
    plt.title("Downstream Accuracy Gap: Oracle vs. MSPR Routing", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(K_vals)
    plt.ylim(0, 100)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("accuracy_vs_k.png", dpi=300)
    plt.close()
    print("Saved accuracy_vs_k.png")

if __name__ == "__main__":
    generate_plots()
