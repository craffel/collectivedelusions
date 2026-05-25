import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_plot():
    results_path = "results.json"
    if not os.path.exists(results_path):
        print("results.json not found yet.")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    methods = ["Static", "PROTO-TTMM", "DF-Bayes-TTMM", "KT-Fisher", "FDF-DPA", "FDF-DPA (Auto)"]
    segments = ["mnist", "kmnist", "fashion", "overall"]
    segment_names = ["MNIST Segment", "KMNIST Segment", "FashionMNIST (Novel)", "Overall Stream"]
    
    # Extract data
    data = {m: [results[m][s] for s in segments] for m in methods}
    
    # Set up plot
    x = np.arange(len(segment_names))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    colors = {
        "Static": "#95a5a6",
        "PROTO-TTMM": "#e74c3c",
        "DF-Bayes-TTMM": "#e67e22",
        "KT-Fisher": "#3498db",
        "FDF-DPA": "#2ecc71",
        "FDF-DPA (Auto)": "#1abc9c"
    }
    
    rects1 = ax.bar(x - 2.5*width, data["Static"], width, label="Static Merging", color=colors["Static"], edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x - 1.5*width, data["PROTO-TTMM"], width, label="PROTO-TTMM (Offline)", color=colors["PROTO-TTMM"], edgecolor="black", linewidth=0.8)
    rects3 = ax.bar(x - 0.5*width, data["DF-Bayes-TTMM"], width, label="DF-Bayes-TTMM (Data-Free)", color=colors["DF-Bayes-TTMM"], edgecolor="black", linewidth=0.8)
    rects4 = ax.bar(x + 0.5*width, data["KT-Fisher"], width, label="KT-Fisher (SOTA, Offline)", color=colors["KT-Fisher"], edgecolor="black", linewidth=0.8)
    rects5 = ax.bar(x + 1.5*width, data["FDF-DPA"], width, label="FDF-DPA (Ours, Heuristic)", color=colors["FDF-DPA"], edgecolor="black", linewidth=0.8)
    rects6 = ax.bar(x + 2.5*width, data["FDF-DPA (Auto)"], width, label="FDF-DPA (Ours, Auto-Threshold)", color=colors["FDF-DPA (Auto)"], edgecolor="black", linewidth=0.8)
    
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Test-Time Model Merging Performance Comparison on Open-World Streams", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(segment_names, fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="lower left")
    
    # Add value labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)
    
    plt.tight_layout()
    plt.savefig("results_plot.png")
    print("Successfully generated and saved results_plot.png!")

if __name__ == "__main__":
    generate_plot()
