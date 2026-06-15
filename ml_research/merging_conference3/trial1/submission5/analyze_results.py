import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    if not os.path.exists('sweep_results.json'):
        print("sweep_results.json not found yet. The sweeps job might still be running.")
        return
        
    with open('sweep_results.json', 'r') as f:
        results = json.load(f)
        
    print(f"Loaded {len(results)} results.")
    
    # Organize results by method
    method_data = {}
    for r in results:
        m = r['method']
        if m not in method_data:
            method_data[m] = []
        method_data[m].append(r)
        
    # 1. Print Summary Table
    print("\n--- Model Merging Comparison ---")
    print(f"{'Method':<35} | {'Best Average Acc':<15} | {'Best Scaling s':<12} | {'Optimal Gamma':<15}")
    print("-" * 85)
    
    table_rows = []
    
    # Task Arithmetic
    ta_runs = method_data.get('Task Arithmetic', [])
    if ta_runs:
        best_ta = max(ta_runs, key=lambda x: x['average_accuracy'])
        print(f"{'Task Arithmetic':<35} | {best_ta['average_accuracy']*100:14.2f}% | {best_ta['scaling']:12.2f} | {'N/A':<15}")
        table_rows.append({
            'method': 'Task Arithmetic',
            'acc': best_ta['average_accuracy'],
            's': best_ta['scaling'],
            'gamma': 'N/A',
            'decoupling': 'none'
        })
        
    # Pure Isotropic Merging
    iso_runs = method_data.get('Pure Isotropic Merging', [])
    if iso_runs:
        best_iso = max(iso_runs, key=lambda x: x['average_accuracy'])
        print(f"{'Pure Isotropic Merging (SAIM)':<35} | {best_iso['average_accuracy']*100:14.2f}% | {best_iso['scaling']:12.2f} | {best_iso['gamma']:15.2f}")
        table_rows.append({
            'method': 'Pure Isotropic Merging (SAIM)',
            'acc': best_iso['average_accuracy'],
            's': best_iso['scaling'],
            'gamma': best_iso['gamma'],
            'decoupling': 'none'
        })
        
    # OrthoMerge (Global)
    om_g_runs = method_data.get('OrthoMerge (global)', [])
    if om_g_runs:
        best_om_g = max(om_g_runs, key=lambda x: x['average_accuracy'])
        print(f"{'OrthoMerge (Global)':<35} | {best_om_g['average_accuracy']*100:14.2f}% | {best_om_g['scaling']:12.2f} | {'1.0 (Fixed)':<15}")
        table_rows.append({
            'method': 'OrthoMerge (Global)',
            'acc': best_om_g['average_accuracy'],
            's': best_om_g['scaling'],
            'gamma': 1.0,
            'decoupling': 'global'
        })
        
    # OrthoMerge (Conflict-Aware)
    om_c_runs = method_data.get('OrthoMerge (conflict_aware)', [])
    if om_c_runs:
        best_om_c = max(om_c_runs, key=lambda x: x['average_accuracy'])
        print(f"{'OrthoMerge (Conflict-Aware)':<35} | {best_om_c['average_accuracy']*100:14.2f}% | {best_om_c['scaling']:12.2f} | {'1.0 (Fixed)':<15}")
        table_rows.append({
            'method': 'OrthoMerge (Conflict-Aware)',
            'acc': best_om_c['average_accuracy'],
            's': best_om_c['scaling'],
            'gamma': 1.0,
            'decoupling': 'conflict_aware'
        })
        
    # TIES-Merging
    for k in [10, 20, 30]:
        ties_runs = method_data.get(f"TIES-Merging (k={k})", [])
        if ties_runs:
            best_ties = max(ties_runs, key=lambda x: x['average_accuracy'])
            print(f"{f'TIES-Merging (k={k})':<35} | {best_ties['average_accuracy']*100:14.2f}% | {best_ties['scaling']:12.2f} | {'N/A':<15}")
            table_rows.append({
                'method': f'TIES-Merging (k={k})',
                'acc': best_ties['average_accuracy'],
                's': best_ties['scaling'],
                'gamma': 'N/A',
                'decoupling': 'none'
            })
            
    # ORIM (Global)
    orim_g_runs = method_data.get('ORIM (global)', [])
    if orim_g_runs:
        best_orim_g = max(orim_g_runs, key=lambda x: x['average_accuracy'])
        print(f"{'ORIM (Global)':<35} | {best_orim_g['average_accuracy']*100:14.2f}% | {best_orim_g['scaling']:12.2f} | {best_orim_g['gamma']:15.2f}")
        table_rows.append({
            'method': 'ORIM (Global)',
            'acc': best_orim_g['average_accuracy'],
            's': best_orim_g['scaling'],
            'gamma': best_orim_g['gamma'],
            'decoupling': 'global'
        })
        
    # ORIM (Conflict-Aware)
    orim_c_runs = method_data.get('ORIM (conflict_aware)', [])
    if orim_c_runs:
        best_orim_c = max(orim_c_runs, key=lambda x: x['average_accuracy'])
        print(f"{'ORIM (Conflict-Aware)':<35} | {best_orim_c['average_accuracy']*100:14.2f}% | {best_orim_c['scaling']:12.2f} | {best_orim_c['gamma']:15.2f}")
        table_rows.append({
            'method': 'ORIM (Conflict-Aware)',
            'acc': best_orim_c['average_accuracy'],
            's': best_orim_c['scaling'],
            'gamma': best_orim_c['gamma'],
            'decoupling': 'conflict_aware'
        })
        
    print("-" * 85)
    
    # 2. Write markdown table
    os.makedirs('results', exist_ok=True)
    with open('results/comparison_table.md', 'w') as f:
        f.write("# Model Merging Evaluation Results\n\n")
        f.write("| Method | Best Average Accuracy | Best Scaling s | Optimal Gamma |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for row in table_rows:
            f.write(f"| {row['method']} | {row['acc']*100:.2f}% | {row['s']:.2f} | {row['gamma']} |\n")
            
    # 3. Create Accuracy vs. Gamma Curve for ORIM
    plt.figure(figsize=(10, 6))
    
    # For a fixed scaling s (say s=0.5 or find optimal s for each gamma)
    gammas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] # 1.0 corresponds to OrthoMerge
    
    # Helper to get best accuracy for a specific gamma and decoupling mode
    def get_accs_for_mode(runs_list, om_runs_list):
        accs = []
        for g in gammas:
            if g == 1.0:
                # OrthoMerge is gamma=1.0
                matching = om_runs_list
            else:
                matching = [r for r in runs_list if abs(r['gamma'] - g) < 1e-5]
            if matching:
                best_match = max(matching, key=lambda x: x['average_accuracy'])
                accs.append(best_match['average_accuracy'] * 100)
            else:
                accs.append(np.nan)
        return accs
        
    orim_g_accs = get_accs_for_mode(orim_g_runs, om_g_runs)
    orim_c_accs = get_accs_for_mode(orim_c_runs, om_c_runs)
    
    plt.plot(gammas, orim_g_accs, marker='o', linestyle='-', linewidth=2, label='ORIM (Global Decoupling)')
    plt.plot(gammas, orim_c_accs, marker='s', linestyle='--', linewidth=2, label='ORIM (Conflict-Aware Decoupling)')
    
    # Add baseline levels
    if ta_runs:
        plt.axhline(y=best_ta['average_accuracy']*100, color='r', linestyle=':', label='Task Arithmetic')
    if iso_runs:
        plt.axhline(y=best_iso['average_accuracy']*100, color='g', linestyle='-.', label='Pure Isotropic (SAIM)')
        
    plt.xlabel('Residual Isotropy Factor (gamma)')
    plt.ylabel('3-Task Average Accuracy (%)')
    plt.title('Performance vs. Residual Isotropy Factor (gamma)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results/accuracy_vs_gamma.png', dpi=300)
    plt.close()
    
    print("Analysis complete! Comparison table written to results/comparison_table.md and plot saved to results/accuracy_vs_gamma.png")

if __name__ == '__main__':
    main()
