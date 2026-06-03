import os
import matplotlib.pyplot as plt

def generate_tradeoff_plot():
    # Data from our empirical sweep
    c_vals = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    wrsa_accs = [42.02, 42.83, 45.29, 51.73, 54.13, 46.21, 22.29]
    
    uncal_avg = 48.06
    sp_taac_avg = 54.17
    fdsa_avg = 41.99
    
    # Create plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(c_vals, wrsa_accs, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='WRSA (Ours)')
    
    # Baselines as horizontal lines
    plt.axhline(y=sp_taac_avg, color='#2ca02c', linestyle='--', linewidth=2, label='SP-TAAC (Spatial)')
    plt.axhline(y=uncal_avg, color='#ff7f0e', linestyle='-.', linewidth=2, label='Uncalibrated WA')
    plt.axhline(y=fdsa_avg, color='#d62728', linestyle=':', linewidth=2, label='FDSA (Naive Spectral)')
    
    # Annotations for different phases
    plt.annotate('Spectral Sparsity Trap\n(Noise Amplification)', 
                 xy=(0.05, 42.5), 
                 xytext=(0.08, 30.0),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, ha='center')
                 
    plt.annotate('Signal Restoration\n(Optimal Trade-off)', 
                 xy=(0.30, 53.0), 
                 xytext=(0.28, 43.0),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, ha='center')
                 
    plt.annotate('Over-regularization\n(Signal Collapse)', 
                 xy=(0.48, 25.0), 
                 xytext=(0.44, 15.0),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, ha='center')
    
    plt.title('WRSA Performance under varying Regularization ($c$)', fontsize=12, fontweight='bold')
    plt.xlabel('Wiener Regularization Parameter ($c$)', fontsize=11)
    plt.ylabel('Average Multi-Task Accuracy (%)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', frameon=True, shadow=False)
    plt.xlim(-0.02, 0.55)
    plt.ylim(5, 60)
    plt.tight_layout()
    
    # Save directory
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/wrsa_tradeoff.pdf', format='pdf', dpi=300)
    plt.savefig('plots/wrsa_tradeoff.png', format='png', dpi=300)
    print("Successfully generated plots/wrsa_tradeoff.pdf and plots/wrsa_tradeoff.png")

if __name__ == '__main__':
    generate_tradeoff_plot()
