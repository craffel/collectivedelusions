import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def set_style():
    # Use clean, professional style parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'

def plot_lr_robustness(df):
    set_style()
    plt.figure(figsize=(5.5, 4))
    
    etas = [0.001, 0.01, 0.1, 1.0]
    
    # Standard TTA (SGD)
    sub_tta = df[(df['method'] == 'Standard TTA') & (df['opt'] == 'sgd')]
    tta_accs = [sub_tta[sub_tta['eta'] == e]['accuracy'].mean() for e in etas]
    
    # FGS-TTA (P=50%, SGD)
    sub_fgs = df[(df['method'] == 'FGS-TTA') & (df['sparsity_p'] == 50) & (df['opt'] == 'sgd')]
    fgs_accs = [sub_fgs[sub_fgs['eta'] == e]['accuracy'].mean() for e in etas]
    
    plt.plot(etas, tta_accs, marker='o', color='#d62728', linestyle='--', linewidth=2, label='Standard TTA (P=0%)')
    plt.plot(etas, fgs_accs, marker='s', color='#1f77b4', linestyle='-', linewidth=2, label='FGS-TTA (P=50%)')
    
    plt.xscale('log')
    plt.xticks(etas, ['0.001', '0.01', '0.1', '1.0'])
    plt.xlabel('Test-Time Learning Rate ($\eta$)')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Robustness to Test-Time Learning Rate', fontsize=12, fontweight='bold', pad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='lower left')
    plt.tight_layout()
    
    plt.savefig('lr_robustness.pdf', format='pdf', dpi=300)
    plt.savefig('lr_robustness.png', format='png', dpi=300)
    print("Saved lr_robustness.pdf and .png")
    plt.close()

def plot_sparsity_impact(df):
    set_style()
    plt.figure(figsize=(5.5, 4))
    
    sparsity_p = [20, 50, 80]
    
    # FGS-TTA (SGD, eta=1.0)
    sub_tta_p = df[(df['method'] == 'FGS-TTA') & (df['eta'] == 1.0) & (df['opt'] == 'sgd')]
    tta_p_accs = [sub_tta_p[sub_tta_p['sparsity_p'] == p]['accuracy'].mean() for p in sparsity_p]
    
    # FGS-LFWA (SGD, eta=0.001, alpha=1.0)
    sub_lfwa_p = df[(df['method'] == 'FGS-LFWA') & (df['eta'] == 0.001) & (df['alpha'] == 1.0) & (df['opt'] == 'sgd')]
    lfwa_p_accs = [sub_lfwa_p[sub_lfwa_p['sparsity_p'] == p]['accuracy'].mean() for p in sparsity_p]
    
    # Add Standard TTA (P=0) for reference at respective optimal learning rates
    # Standard TTA (SGD, eta=0.01)
    tta_p0 = df[(df['method'] == 'Standard TTA') & (df['eta'] == 0.01) & (df['opt'] == 'sgd')]['accuracy'].mean()
    # LFWA (SGD, eta=0.01, alpha=0.5)
    lfwa_p0 = df[(df['method'] == 'LFWA') & (df['eta'] == 0.01) & (df['alpha'] == 0.5) & (df['opt'] == 'sgd')]['accuracy'].mean()
    
    all_p = [0] + sparsity_p
    all_tta_accs = [tta_p0] + tta_p_accs
    all_lfwa_accs = [lfwa_p0] + lfwa_p_accs
    
    plt.plot(all_p, all_tta_accs, marker='o', color='#e377c2', linestyle='-', linewidth=2, label='FGS-TTA ($\eta=1.0$)')
    plt.plot(all_p, all_lfwa_accs, marker='^', color='#2ca02c', linestyle='-', linewidth=2, label='FGS-LFWA ($\eta=0.001, \\alpha=1.0$)')
    
    plt.xticks(all_p, ['0% (Standard)', '20%', '50%', '80%'])
    plt.xlabel('Sparsity Threshold ($P\%$)')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Impact of Sparsity Threshold ($P\%$)', fontsize=12, fontweight='bold', pad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    
    plt.savefig('sparsity_impact.pdf', format='pdf', dpi=300)
    plt.savefig('sparsity_impact.png', format='png', dpi=300)
    print("Saved sparsity_impact.pdf and .png")
    plt.close()

def main():
    try:
        df = pd.read_csv('experiment_results.csv')
    except FileNotFoundError:
        print("experiment_results.csv not found!")
        return
        
    plot_lr_robustness(df)
    plot_sparsity_impact(df)

if __name__ == '__main__':
    main()
