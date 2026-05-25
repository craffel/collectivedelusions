import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Since we need to wait for results to be ready, we'll wait for them
    results_dir = "results"
    methods = ["Static", "PROTO-TTMM", "IGGS-OW", "Rob-OW"]
    corruptions = ["clean", "corrupted"]
    
    all_files = []
    for method in methods:
        for corr in corruptions:
            all_files.append(f"result_{method}_{corr}_seed42.txt")
            
    print("Waiting for results files to generate plots...")
    while True:
        missing = [f for f in all_files if not os.path.exists(os.path.join(results_dir, f))]
        if not missing:
            break
        import time
        time.sleep(10)
        
    # Read results
    data = {}
    for method in methods:
        data[method] = {}
        for corr in corruptions:
            filepath = os.path.join(results_dir, f"result_{method}_{corr}_seed42.txt")
            data[method][corr] = {}
            with open(filepath, "r") as f:
                content = f.read()
            for line in content.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if k in ["MNIST_Acc", "KMNIST_Acc", "FashionMNIST_Acc", "Overall_Acc", "NDR", "FPR"]:
                        data[method][corr][k] = float(v)

    # Set up styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    # Extract values for the plot (Overall stream accuracy)
    clean_accs = [data[m]['clean']['Overall_Acc'] for m in methods]
    corr_accs = [data[m]['corrupted']['Overall_Acc'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Create bars with distinct, professional colors
    rects1 = ax.bar(x - width/2, clean_accs, width, label='Clean Stream', color='#4C72B0', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, corr_accs, width, label='Corrupted Stream (σ=0.2)', color='#DD8452', edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Overall Stream Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Overall Stream Classification Accuracy Comparison\n(MNIST $\\rightarrow$ KMNIST $\\rightarrow$ FashionMNIST)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor='white', framealpha=1.0, edgecolor='lightgray', fontsize=10)
    
    # Helper to add value labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('results_chart.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Successfully generated and saved results_chart.pdf (Overall Accuracy)!")

if __name__ == "__main__":
    main()
