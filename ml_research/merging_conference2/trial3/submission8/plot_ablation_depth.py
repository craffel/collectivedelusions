import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    ablation_path = 'results/ablation_depth.json'
    if not os.path.exists(ablation_path):
        print(f"Ablation results not found at {ablation_path}")
        return
        
    with open(ablation_path, 'r') as f:
        data = json.load(f)
        
    groups = ['none', 'shallow', 'middle', 'deep', 'all']
    group_labels = ['None\n(Uncalibrated)', 'Shallow\n(bn1 + layer1)', 'Middle\n(layer2 + layer3)', 'Deep\n(layer4)', 'All Layers\n(SP-TAAC)']
    
    mnist = [data[g]['mnist'] for g in groups]
    fashion = [data[g]['fashion'] for g in groups]
    cifar = [data[g]['cifar'] for g in groups]
    avg = [data[g]['avg'] for g in groups]
    
    x = np.arange(len(groups))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Elegant, publication-quality color scheme
    rects1 = ax.bar(x - 1.5*width, mnist, width, label='MNIST', color='#34495e')
    rects2 = ax.bar(x - 0.5*width, fashion, width, label='Fashion-MNIST', color='#3498db')
    rects3 = ax.bar(x + 0.5*width, cifar, width, label='CIFAR-10', color='#e74c3c')
    rects4 = ax.bar(x + 1.5*width, avg, width, label='Average', color='#2ecc71', hatch='//')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('SP-TAAC Layer Calibration Depth Ablation Study', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    ax.legend(fontsize=11, loc='upper left', frameon=True, framealpha=0.9, edgecolor='#bdc3c7')
    
    # Add values on top of the average bars
    for i, val in enumerate(avg):
        ax.annotate(f"{val:.1f}%",
                    xy=(x[i] + 1.5*width, val),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plot_out = 'results/ablation_depth.png'
    plt.savefig(plot_out, dpi=100)
    plt.close()
    
    # Also save as PDF for LaTeX compatibility (vector graphics)
    try:
        from PIL import Image
        fig, ax = plt.subplots(figsize=(10, 6.5))
        rects1 = ax.bar(x - 1.5*width, mnist, width, label='MNIST', color='#34495e')
        rects2 = ax.bar(x - 0.5*width, fashion, width, label='Fashion-MNIST', color='#3498db')
        rects3 = ax.bar(x + 0.5*width, cifar, width, label='CIFAR-10', color='#e74c3c')
        rects4 = ax.bar(x + 1.5*width, avg, width, label='Average', color='#2ecc71', hatch='//')
        ax.set_ylabel('Test Accuracy (%)', fontsize=13)
        ax.set_title('SP-TAAC Layer Calibration Depth Ablation Study', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=11, loc='upper left', frameon=True, framealpha=0.9, edgecolor='#bdc3c7')
        for i, val in enumerate(avg):
            ax.annotate(f"{val:.1f}%",
                        xy=(x[i] + 1.5*width, val),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        pdf_out = 'ablation_depth.pdf'
        plt.savefig(pdf_out)
        plt.savefig('results/ablation_depth.pdf')
        plt.close()
        print(f"Saved PDF ablation plot to {pdf_out}")
    except Exception as e:
        print(f"Could not save PDF version: {e}")
        
    print(f"Saved PNG ablation plot to {plot_out}")

if __name__ == '__main__':
    main()
