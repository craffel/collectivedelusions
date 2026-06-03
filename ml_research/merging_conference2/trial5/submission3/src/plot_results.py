import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Loading evaluation results...")
    try:
        data = np.load('evaluation_results.npz', allow_pickle=True)
    except FileNotFoundError:
        print("evaluation_results.npz not found. Please run evaluate.py first.")
        return

    # Extract data with backward-compatible fallbacks
    p_levels = data['p_levels']
    sweep1 = data['sweep1_results'].item()
    sweep1_stds = data['sweep1_stds'].item() if 'sweep1_stds' in data.files else {m: [0.0]*len(p_levels) for m in sweep1.keys()}
    
    n_budgets = data['n_budgets']
    sweep2 = data['sweep2_results'].item()
    sweep2_stds = data['sweep2_stds'].item() if 'sweep2_stds' in data.files else {m: [0.0]*len(n_budgets) for m in sweep2.keys()}
    
    ablation = data['ablation_results'].item()
    ablation_stds = data['ablation_stds'].item() if 'ablation_stds' in data.files else {m: 0.0 for m in ablation.keys()}
    
    ta_results = data['ta_results'].item()

    # Set up plotting style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

    # --- Plot 1: Corruption and Budget Panels ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Outlier Corruption Sweep
    ax = axes[0]
    ax.plot(p_levels, [x for x in sweep1['none']], label='Uncalibrated (WA)', color='#7f7f7f', linestyle='--', marker='o')
    if 'none' in sweep1_stds:
        ax.fill_between(p_levels, np.array(sweep1['none']) - np.array(sweep1_stds['none']), np.array(sweep1['none']) + np.array(sweep1_stds['none']), color='#7f7f7f', alpha=0.1)

    ax.plot(p_levels, [x for x in sweep1['sp-taac']], label='SP-TAAC (Global)', color='#bcbd22', marker='v')
    if 'sp-taac' in sweep1_stds:
        ax.fill_between(p_levels, np.array(sweep1['sp-taac']) - np.array(sweep1_stds['sp-taac']), np.array(sweep1['sp-taac']) + np.array(sweep1_stds['sp-taac']), color='#bcbd22', alpha=0.1)

    ax.plot(p_levels, [x for x in sweep1['taac']], label='Standard TAAC', color='#d62728', marker='s')
    if 'taac' in sweep1_stds:
        ax.fill_between(p_levels, np.array(sweep1['taac']) - np.array(sweep1_stds['taac']), np.array(sweep1['taac']) + np.array(sweep1_stds['taac']), color='#d62728', alpha=0.1)

    ax.plot(p_levels, [x for x in sweep1['slf-taac']], label='SLF-TAAC (Baseline)', color='#1f77b4', marker='x')
    if 'slf-taac' in sweep1_stds:
        ax.fill_between(p_levels, np.array(sweep1['slf-taac']) - np.array(sweep1_stds['slf-taac']), np.array(sweep1['slf-taac']) + np.array(sweep1_stds['slf-taac']), color='#1f77b4', alpha=0.1)

    ax.plot(p_levels, [x for x in sweep1['qrc']], label='Proposed QRC (Ours)', color='#2ca02c', linewidth=2.5, marker='*')
    if 'qrc' in sweep1_stds:
        ax.fill_between(p_levels, np.array(sweep1['qrc']) - np.array(sweep1_stds['qrc']), np.array(sweep1['qrc']) + np.array(sweep1_stds['qrc']), color='#2ca02c', alpha=0.15)
    
    ax.set_xlabel('Outlier Corruption Fraction ($p$)')
    ax.set_ylabel('Average Multi-Task Accuracy (%)')
    ax.set_title('(a) Robustness under Calibration Outliers ($N=128$)')
    ax.set_ylim(10, 65)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left')

    # Panel B: Sample Budget Sweep
    ax = axes[1]
    ax.plot(n_budgets, [x for x in sweep2['taac']], label='Standard TAAC', color='#d62728', marker='s')
    if 'taac' in sweep2_stds:
        ax.fill_between(n_budgets, np.array(sweep2['taac']) - np.array(sweep2_stds['taac']), np.array(sweep2['taac']) + np.array(sweep2_stds['taac']), color='#d62728', alpha=0.1)

    ax.plot(n_budgets, [x for x in sweep2['slf-taac']], label='SLF-TAAC', color='#1f77b4', marker='x')
    if 'slf-taac' in sweep2_stds:
        ax.fill_between(n_budgets, np.array(sweep2['slf-taac']) - np.array(sweep2_stds['slf-taac']), np.array(sweep2['slf-taac']) + np.array(sweep2_stds['slf-taac']), color='#1f77b4', alpha=0.1)

    ax.plot(n_budgets, [x for x in sweep2['qrc']], label='Proposed QRC (Ours)', color='#2ca02c', linewidth=2.5, marker='*')
    if 'qrc' in sweep2_stds:
        ax.fill_between(n_budgets, np.array(sweep2['qrc']) - np.array(sweep2_stds['qrc']), np.array(sweep2['qrc']) + np.array(sweep2_stds['qrc']), color='#2ca02c', alpha=0.15)
    
    ax.set_xlabel('Calibration Sample Budget ($N$ per task)')
    ax.set_ylabel('Average Multi-Task Accuracy (%)')
    ax.set_title('(b) Sample Efficiency on Clean Data ($p=0.0$)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(n_budgets)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(10, 65)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('fig_corruption_budget.png', dpi=300)
    plt.savefig('template/fig_corruption_budget.png', dpi=300)
    plt.close()
    print("Saved fig_corruption_budget.png to root and template/")

    # --- Plot 2: Ablation Bar Chart ---
    plt.figure(figsize=(6, 4))
    methods_abl = ['Uncalibrated', 'Standard TAAC', 'Median-Only', 'IQR-Only', 'QRC (Median + IQR)']
    scores_abl = [
        ablation['none'],
        ablation['taac'],
        ablation['qrc-median'],
        ablation['qrc-iqr'],
        ablation['qrc']
    ]
    stds_abl = [
        ablation_stds.get('none', 0.0),
        ablation_stds.get('taac', 0.0),
        ablation_stds.get('qrc-median', 0.0),
        ablation_stds.get('qrc-iqr', 0.0),
        ablation_stds.get('qrc', 0.0)
    ]
    colors = ['#7f7f7f', '#d62728', '#aec7e8', '#ffbb78', '#2ca02c']
    
    bars = plt.bar(methods_abl, scores_abl, yerr=stds_abl, capsize=5, color=colors, width=0.6, edgecolor='black', linewidth=0.7)
    plt.ylabel('Average Multi-Task Accuracy (%)')
    plt.title('Ablation of QRC Components ($p=0.2$, $N=128$)')
    plt.ylim(10, 60)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig('fig_ablation.png', dpi=300)
    plt.savefig('template/fig_ablation.png', dpi=300)
    plt.close()
    print("Saved fig_ablation.png to root and template/")

if __name__ == '__main__':
    main()
