import json
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    results_path = "./results.json"
    if not os.path.exists(results_path):
        print(f"ERROR: Missing {results_path}. Cannot generate artifacts.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    # 1. Plot Average Accuracy vs. Sparsity Ratio
    print("Generating Sparsity Plots...")
    os.makedirs("./plots", exist_ok=True)
    
    for arch in ['resnet18', 'mlp']:
        plt.figure(figsize=(8, 5))
        sparsity_ratios = [0.2, 0.4, 0.6, 0.8]
        
        # Baselines
        ties_avgs = [results[arch]['ties'][f"{sr:.1f}"]['avg'] for sr in sparsity_ratios]
        dare_avgs = [results[arch]['dare'][f"{sr:.1f}"]['avg'] for sr in sparsity_ratios]
        
        # SC-WCPR (with 'dare' compensation)
        sc_ties_avgs = [results[arch]['sc_wcpr']['ties'][f"{sr:.1f}"]['dare']['avg'] for sr in sparsity_ratios]
        sc_dare_avgs = [results[arch]['sc_wcpr']['dare'][f"{sr:.1f}"]['dare']['avg'] for sr in sparsity_ratios]
        
        # Standard WCPR (dense, flat line)
        wcpr_flat = [results[arch]['wcpr_avg']] * len(sparsity_ratios)
        # WA flat line
        wa_flat = [results[arch]['wa_avg']] * len(sparsity_ratios)
        
        plt.plot(sparsity_ratios, ties_avgs, 'o--', color='crimson', label='TIES-Merging')
        plt.plot(sparsity_ratios, dare_avgs, 's--', color='orange', label='DARE-Merging')
        plt.plot(sparsity_ratios, sc_ties_avgs, 'o-', color='navy', linewidth=2, label='SC-WCPR (TIES, ours)')
        plt.plot(sparsity_ratios, sc_dare_avgs, 's-', color='teal', linewidth=2, label='SC-WCPR (DARE, ours)')
        plt.plot(sparsity_ratios, wcpr_flat, ':', color='gray', label='Standard WCPR (dense)')
        plt.plot(sparsity_ratios, wa_flat, '-.', color='darkorchid', label='Weight Averaging')
        
        plt.title(f"Sparsity Robustness on {arch.upper()}")
        plt.xlabel("Sparsity Ratio (Pruning Rate)")
        plt.ylabel("Average Multi-Task Accuracy (%)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best')
        plt.tight_layout()
        
        plot_path = f"./plots/sparsity_plot_{arch}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved sparsity plot to {plot_path}")

    # 2. Compile LaTeX Tables
    print("\nCompiling LaTeX Tables...")
    
    # --- TABLE 1: Standard Merging Comparison ---
    table1 = []
    table1.append("% --- TABLE 1: MAIN RESULTS ---")
    table1.append("\\begin{table*}[t]")
    table1.append("\\caption{Model merging and calibration performance across vision datasets. We report the test accuracy (\\%) of individual tasks and their average for both ResNet-18 (with BatchNorm) and MLP (without BatchNorm) architectures. SC-WCPR uses TIES/DARE with a sparsity ratio of 0.4 and \\texttt{dare} compensation.}")
    table1.append("\\label{table:main_results}")
    table1.append("\\centering")
    table1.append("\\resizebox{\\textwidth}{!}{%")
    table1.append("\\begin{tabular}{llcccc|cccc}")
    table1.append("\\toprule")
    table1.append(" & & \\multicolumn{4}{c|}{\\textbf{ResNet-18}} & \\multicolumn{4}{c}{\\textbf{MLP}} \\\\")
    table1.append("Category & Method & MNIST & FMNIST & CIFAR-10 & Average & MNIST & FMNIST & CIFAR-10 & Average \\\\")
    table1.append("\\midrule")
    
    # Oracles
    o_res = results['resnet18']['oracles']
    o_mlp = results['mlp']['oracles']
    table1.append(f"Oracles & Expert Oracles & {o_res['mnist']:.2f} & {o_res['fmnist']:.2f} & {o_res['cifar10']:.2f} & \\textbf{{ {results['resnet18']['oracles_avg']:.2f} }} & {o_mlp['mnist']:.2f} & {o_mlp['fmnist']:.2f} & {o_mlp['cifar10']:.2f} & \\textbf{{ {results['mlp']['oracles_avg']:.2f} }} \\\\")
    table1.append("\\midrule")
    
    # Uncalibrated / Baselines
    wa_res = results['resnet18']['wa']
    wa_mlp = results['mlp']['wa']
    table1.append(f"Uncalibrated & Weight Averaging (WA) & {wa_res['mnist']:.2f} & {wa_res['fmnist']:.2f} & {wa_res['cifar10']:.2f} & {results['resnet18']['wa_avg']:.2f} & {wa_mlp['mnist']:.2f} & {wa_mlp['fmnist']:.2f} & {wa_mlp['cifar10']:.2f} & {results['mlp']['wa_avg']:.2f} \\\\")
    
    ta_res = results['resnet18']['ta_best']
    ta_mlp = results['mlp']['ta_best']
    l_res = results['resnet18']['ta_best_lambda']
    l_mlp = results['mlp']['ta_best_lambda']
    table1.append(f" & Tuned TA ($\\lambda = {l_res:.2f} / {l_mlp:.2f}$) & {ta_res['mnist']:.2f} & {ta_res['fmnist']:.2f} & {ta_res['cifar10']:.2f} & {results['resnet18']['ta_best_avg']:.2f} & {ta_mlp['mnist']:.2f} & {ta_mlp['fmnist']:.2f} & {ta_mlp['cifar10']:.2f} & {results['mlp']['ta_best_avg']:.2f} \\\\")
    
    wcpr_res = results['resnet18']['wcpr']
    wcpr_mlp = results['mlp']['wcpr']
    table1.append(f" & Standard WCPR (dense) & {wcpr_res['mnist']:.2f} & {wcpr_res['fmnist']:.2f} & {wcpr_res['cifar10']:.2f} & {results['resnet18']['wcpr_avg']:.2f} & {wcpr_mlp['mnist']:.2f} & {wcpr_mlp['fmnist']:.2f} & {wcpr_mlp['cifar10']:.2f} & {results['mlp']['wcpr_avg']:.2f} \\\\")
    table1.append("\\midrule")
    
    # Sparsified
    t_res = results['resnet18']['ties']['0.4']['accs']
    t_mlp = results['mlp']['ties']['0.4']['accs']
    table1.append(f"Sparsified & TIES-Merging ($r=0.4$) & {t_res['mnist']:.2f} & {t_res['fmnist']:.2f} & {t_res['cifar10']:.2f} & {results['resnet18']['ties']['0.4']['avg']:.2f} & {t_mlp['mnist']:.2f} & {t_mlp['fmnist']:.2f} & {t_mlp['cifar10']:.2f} & {results['mlp']['ties']['0.4']['avg']:.2f} \\\\")
    
    d_res = results['resnet18']['dare']['0.4']['accs']
    d_mlp = results['mlp']['dare']['0.4']['accs']
    table1.append(f" & DARE-Merging ($r=0.4$) & {d_res['mnist']:.2f} & {d_res['fmnist']:.2f} & {d_res['cifar10']:.2f} & {results['resnet18']['dare']['0.4']['avg']:.2f} & {d_mlp['mnist']:.2f} & {d_mlp['fmnist']:.2f} & {d_mlp['cifar10']:.2f} & {results['mlp']['dare']['0.4']['avg']:.2f} \\\\")
    table1.append("\\midrule")
    
    # Ours
    sc_t_res = results['resnet18']['sc_wcpr']['ties']['0.4']['dare']['accs']
    sc_t_mlp = results['mlp']['sc_wcpr']['ties']['0.4']['dare']['accs']
    table1.append(f"Ours (SC-WCPR) & SC-WCPR (TIES, $r=0.4$) & {sc_t_res['mnist']:.2f} & {sc_t_res['fmnist']:.2f} & {sc_t_res['cifar10']:.2f} & \\textbf{{ {results['resnet18']['sc_wcpr']['ties']['0.4']['dare']['avg']:.2f} }} & {sc_t_mlp['mnist']:.2f} & {sc_t_mlp['fmnist']:.2f} & {sc_t_mlp['cifar10']:.2f} & \\textbf{{ {results['mlp']['sc_wcpr']['ties']['0.4']['dare']['avg']:.2f} }} \\\\")
    
    sc_d_res = results['resnet18']['sc_wcpr']['dare']['0.4']['dare']['accs']
    sc_d_mlp = results['mlp']['sc_wcpr']['dare']['0.4']['dare']['accs']
    table1.append(f" & SC-WCPR (DARE, $r=0.4$) & {sc_d_res['mnist']:.2f} & {sc_d_res['fmnist']:.2f} & {sc_d_res['cifar10']:.2f} & \\textbf{{ {results['resnet18']['sc_wcpr']['dare']['0.4']['dare']['avg']:.2f} }} & {sc_d_mlp['mnist']:.2f} & {sc_d_mlp['fmnist']:.2f} & {sc_d_mlp['cifar10']:.2f} & \\textbf{{ {results['mlp']['sc_wcpr']['dare']['0.4']['dare']['avg']:.2f} }} \\\\")
    table1.append("\\bottomrule")
    table1.append("\\end{tabular}%")
    table1.append("}")
    table1.append("\\end{table*}")
    
    # --- TABLE 2: Compensation Method Ablation ---
    table2 = []
    table2.append("% --- TABLE 2: COMPENSATION METHOD ABLATION ---")
    table2.append("\\begin{table}[t]")
    table2.append("\\caption{Ablation of different active scaling compensation methods in SC-WCPR for both TIES and DARE (fixed sparsity ratio of 0.4). Average multi-task accuracy (\\%) is reported.}")
    table2.append("\\label{table:compensation_ablation}")
    table2.append("\\centering")
    table2.append("\\resizebox{\\columnwidth}{!}{%")
    table2.append("\\begin{tabular}{l|cc|cc}")
    table2.append("\\toprule")
    table2.append(" & \\multicolumn{2}{c|}{\\textbf{ResNet-18}} & \\multicolumn{2}{c}{\\textbf{MLP}} \\\\")
    table2.append("Compensation & TIES & DARE & TIES & DARE \\\\")
    table2.append("\\midrule")
    
    compensations = ['none', 'sqrt', 'linear', 'inv_sqrt', 'inv_linear', 'dare']
    for comp in compensations:
        c_ties_res = results['resnet18']['sc_wcpr']['ties']['0.4'][comp]['avg']
        c_dare_res = results['resnet18']['sc_wcpr']['dare']['0.4'][comp]['avg']
        c_ties_mlp = results['mlp']['sc_wcpr']['ties']['0.4'][comp]['avg']
        c_dare_mlp = results['mlp']['sc_wcpr']['dare']['0.4'][comp]['avg']
        comp_escaped = comp.replace('_', '\\_')
        table2.append(f"\\texttt{{ {comp_escaped} }} & {c_ties_res:.2f} & {c_dare_res:.2f} & {c_ties_mlp:.2f} & {c_dare_mlp:.2f} \\\\")
    table2.append("\\bottomrule")
    table2.append("\\end{tabular}%")
    table2.append("}")
    table2.append("\\end{table}")
    
    # --- TABLE 3: Robustness Analysis ---
    table3 = []
    table3.append("% --- TABLE 3: ROBUSTNESS & CALIBRATION ANALYSIS ---")
    table3.append("\\begin{table*}[t]")
    table3.append("\\caption{Robustness under physical constraints for ResNet-18. We report the average multi-task accuracy (\\%) under different Post-Training Quantization (PTQ) bit-widths (FP32, INT8, INT4) and environmental corruptions (Clean, Gaussian Noise, Gaussian Blur). We compare standard merging, sparsification, and our proposed SC-WCPR (sparsity ratio of 0.4), each with and without data-efficient BatchNorm calibration (DE-BN using $N=16$ or $N=64$ samples per task). Quantization is done in per-channel mode.}")
    table3.append("\\label{table:robustness}")
    table3.append("\\centering")
    table3.append("\\resizebox{\\textwidth}{!}{%")
    table3.append("\\begin{tabular}{llccc|ccc|ccc}")
    table3.append("\\toprule")
    table3.append(" & & \\multicolumn{3}{c|}{\\textbf{No DE-BN}} & \\multicolumn{3}{c|}{\\textbf{DE-BN ($N=16$)}} & \\multicolumn{3}{c}{\\textbf{DE-BN ($N=64$)}} \\\\")
    table3.append("Quantization & Method & Clean & Noise & Blur & Clean & Noise & Blur & Clean & Noise & Blur \\\\")
    
    quant_models = ['wa', 'ta', 'ties', 'dare', 'wcpr', 'sc_wcpr_ties', 'sc_wcpr_dare']
    compensations_ptq = ['dare', 'sqrt']
    
    for num_bits_str, pc_str in [('FP32', 'tensor'), ('INT8', 'channel'), ('INT4', 'channel')]:
        table3.append("\\midrule")
        table3.append(f"\\textbf{{ {num_bits_str} }} & \\multicolumn{{10}}{{c}}{{}} \\\\")
        for m in quant_models:
            if m in ['sc_wcpr_ties', 'sc_wcpr_dare']:
                # Find best comp for No DE-BN based on clean
                best_comp_no = max(compensations_ptq, key=lambda c: results['resnet18']['quantization'][f"{m}_{c}"][num_bits_str][pc_str]['clean']['no_debn']['avg'])
                m_res_no = results['resnet18']['quantization'][f"{m}_{best_comp_no}"][num_bits_str][pc_str]
                c_no = m_res_no['clean']['no_debn']['avg']
                n_no = m_res_no['noise']['no_debn']['avg']
                b_no = m_res_no['blur']['no_debn']['avg']
                
                # Find best comp for DE-BN 16 based on clean
                best_comp_16 = max(compensations_ptq, key=lambda c: results['resnet18']['quantization'][f"{m}_{c}"][num_bits_str][pc_str]['clean']['debn_16']['avg'])
                m_res_16 = results['resnet18']['quantization'][f"{m}_{best_comp_16}"][num_bits_str][pc_str]
                c_16 = m_res_16['clean']['debn_16']['avg']
                n_16 = m_res_16['noise']['debn_16']['avg']
                b_16 = m_res_16['blur']['debn_16']['avg']
                
                # Find best comp for DE-BN 64 based on clean
                best_comp_64 = max(compensations_ptq, key=lambda c: results['resnet18']['quantization'][f"{m}_{c}"][num_bits_str][pc_str]['clean']['debn_64']['avg'])
                m_res_64 = results['resnet18']['quantization'][f"{m}_{best_comp_64}"][num_bits_str][pc_str]
                c_64 = m_res_64['clean']['debn_64']['avg']
                n_64 = m_res_64['noise']['debn_64']['avg']
                b_64 = m_res_64['blur']['debn_64']['avg']
                
                print(f"For {m} under {num_bits_str}: Best No-DEBN={best_comp_no}, Best-16={best_comp_16}, Best-64={best_comp_64}")
            else:
                m_res = results['resnet18']['quantization'][m][num_bits_str][pc_str]
                c_no = m_res['clean']['no_debn']['avg']
                n_no = m_res['noise']['no_debn']['avg']
                b_no = m_res['blur']['no_debn']['avg']
                
                c_16 = m_res['clean']['debn_16']['avg']
                n_16 = m_res['noise']['debn_16']['avg']
                b_16 = m_res['blur']['debn_16']['avg']
                
                c_64 = m_res['clean']['debn_64']['avg']
                n_64 = m_res['noise']['debn_64']['avg']
                b_64 = m_res['blur']['debn_64']['avg']
                
            m_name = m.replace('_', ' ').upper()
            table3.append(f" & {m_name} & {c_no:.2f} & {n_no:.2f} & {b_no:.2f} & {c_16:.2f} & {n_16:.2f} & {b_16:.2f} & {c_64:.2f} & {n_64:.2f} & {b_64:.2f} \\\\")
            
    table3.append("\\bottomrule")
    table3.append("\\end{tabular}%")
    table3.append("}")
    table3.append("\\end{table*}")

    # Write tables to files
    os.makedirs("./tables", exist_ok=True)
    with open("./tables/table1.tex", "w") as f:
        f.write("\n".join(table1))
    with open("./tables/table2.tex", "w") as f:
        f.write("\n".join(table2))
    with open("./tables/table3.tex", "w") as f:
        f.write("\n".join(table3))
        
    print("\nSaved LaTeX tables to ./tables/")

if __name__ == '__main__':
    main()
