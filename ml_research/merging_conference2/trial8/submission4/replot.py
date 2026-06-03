import json
import matplotlib.pyplot as plt

def main():
    with open('results.json', 'r') as f:
        res = json.load(f)
        
    expert_accs = res['expert_accs']
    wa_results = res['wa_results']
    uipr_results = res['uipr_results']
    hns_results = res['hns_results']
    ta_results = res['ta_results']
    cpr_results = res['cpr_results']
    
    # Extra baselines
    best_ties_avg = res['best_ties']['avg_acc']
    best_dare_avg = res['best_dare']['avg_acc']
    
    plt.figure(figsize=(10, 6))
    
    # Line for TA
    ta_x = [r['lambda'] for r in ta_results]
    ta_y = [r['avg'] for r in ta_results]
    plt.plot(ta_x, ta_y, marker='o', linestyle='-', color='blue', label='Task Arithmetic (TA)')
    
    # Line for CPR
    cpr_x = [r['c'] for r in cpr_results]
    cpr_y = [r['avg'] for r in cpr_results]
    plt.plot(cpr_x, cpr_y, marker='s', linestyle='--', color='green', label='Constant Parameter Resonance (CPR, Ours)')
    
    # Horizontal lines for baseline models
    plt.axhline(y=wa_results['avg'], color='red', linestyle=':', label='Weight Averaging (WA)')
    plt.axhline(y=uipr_results['avg'], color='purple', linestyle='-.', label='Update-level IPR')
    plt.axhline(y=hns_results['avg'], color='orange', linestyle='-', label='Holographic Norm Scaling (HNS)')
    
    # Add new baselines
    plt.axhline(y=best_ties_avg, color='brown', linestyle='--', label=f'TIES-Merging (Best: {best_ties_avg:.2f}%)')
    plt.axhline(y=best_dare_avg, color='teal', linestyle='-.', label=f'DARE-Merging (Best: {best_dare_avg:.2f}%)')
    
    # Highlight theoretical constant sqrt(K)
    plt.axvline(x=1.732, color='darkgreen', linestyle=':', label=r'Theoretical Attractor $\sqrt{3} \approx 1.732$')
    
    plt.title('Multi-Task Model Merging Performance Comparison')
    plt.xlabel(r'Scaling Factor ($\lambda$ for TA / $c$ for CPR)')
    plt.ylabel('Average Accuracy (%)')
    plt.ylim(10, 75)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.savefig('cpr_vs_baselines.png', dpi=300, bbox_inches='tight')
    print("Successfully regenerated plot cpr_vs_baselines.png!")

if __name__ == '__main__':
    main()
