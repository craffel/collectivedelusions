import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    sweep_path = 'results/rtaac_sweep.json'
    if not os.path.exists(sweep_path):
        print(f"Sweep results not found at {sweep_path}")
        return
        
    with open(sweep_path, 'r') as f:
        data = json.load(f)
        
    wa_data = data.get('wa', {})
    if not wa_data:
        print("No WA data found in R-TAAC sweep results.")
        return
        
    # Calibration sizes and alphas
    ns = ['4', '8', '16', '32', '64', '128', '256']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_strs = [f"{a:.1f}" for a in alphas]
    
    # Colors and markers for the line plot
    # We use a beautiful color gradient (cool/warm or distinct colors)
    colors = {
        '4': '#d35400',    # Red-Orange for extremely small N
        '8': '#e67e22',    # Orange
        '16': '#f1c40f',   # Yellow
        '32': '#2ecc71',   # Light Green
        '64': '#1abc9c',   # Teal
        '128': '#3498db',  # Blue
        '256': '#9b59b6'   # Purple
    }
    markers = {
        '4': 'o',
        '8': 's',
        '16': '^',
        '32': 'v',
        '64': 'D',
        '128': '<',
        '256': '>'
    }
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    
    for n in ns:
        accs = []
        for a_str in alpha_strs:
            accs.append(wa_data[n][a_str]['avg'])
        ax.plot(alphas, accs, label=f'N = {n}', color=colors[n], marker=markers[n], linewidth=2.2, markersize=7)
        
    ax.set_xlabel(r'Shrinkage Parameter ($\alpha$)', fontsize=13)
    ax.set_ylabel('Average Test Accuracy (%)', fontsize=13)
    ax.set_title(r'R-TAAC Calibration: Shrinkage Trade-off ($\alpha$) vs. Calibration Size ($N$)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(30, 70)
    ax.set_xticks(alphas)
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    
    # Add a horizontal line for the baseline (uncalibrated WA is about 35.26%)
    ax.axhline(y=35.26, color='#7f8c8d', linestyle=':', linewidth=1.5, label='Uncalibrated Baseline (35.26%)')
    
    ax.legend(fontsize=10.5, loc='lower right', frameon=True, framealpha=0.9, edgecolor='#bdc3c7')
    
    # Annotate the peak of N=4 to highlight the Stein's shrinkage effect
    ax.annotate('Stein\'s Shrinkage Peak (+4.95% absolute)', 
                xy=(0.7, 55.82), 
                xytext=(0.3, 58),
                arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=8),
                fontsize=11, fontweight='bold', color='#d35400')
                
    plt.tight_layout()
    plot_out = 'results/rtaac_shrinkage.png'
    plt.savefig(plot_out, dpi=100)
    plt.close()
    print(f"Saved R-TAAC shrinkage plot to {plot_out}")

if __name__ == '__main__':
    main()
