import json
import numpy as np

def format_latex_table(results):
    # We want to display results for N in [16, 64, 256]
    # For each N, we show:
    # 1. Uncalibrated (WA) + L2-LSHA (best reg)
    # 2. Uncalibrated (WA) + PR-LSHA (best reg)
    # 3. Calibrated (WFC) + L2-LSHA (best reg)
    # 4. Calibrated (WFC) + PR-LSHA (best reg) -- OUR METHOD
    
    # We also want to find the best reg for each configuration to report the optimal performance
    best_configs = {}
    for entry in results:
        key = (entry['N'], entry['backbone'], entry['head'])
        if key not in best_configs or entry['avg'] > best_configs[key]['avg']:
            best_configs[key] = entry
            
    print("Optimal Configurations found:")
    for key, val in sorted(best_configs.items()):
        print(f"N={key[0]}, Backbone={key[1]}, Head={key[2]} -> Best Reg={val['reg']}, Avg Acc={val['avg']*100:.2f}%")
        
    # Generate LaTeX table body
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\caption{Ablation study of individual components of our proposed joint calibration framework under different calibration budgets ($N$). We compare direct average backbone (WA) vs. weight-folded calibrated backbone (WFC), and standard L2 least-squares head alignment (L2-LSHA) vs. our prior-regularized least-squares head alignment (PR-LSHA). For each combination, we report the accuracy using the optimal regularization parameter $\lambda$.}")
    latex_lines.append(r"\label{tab:ablation_results}")
    latex_lines.append(r"\vskip 0.15in")
    latex_lines.append(r"\begin{center}")
    latex_lines.append(r"\begin{small}")
    latex_lines.append(r"\begin{sc}")
    latex_lines.append(r"\begin{tabular}{llccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Backbone & Head Alignment & Budget ($N$) & Optimal $\lambda$ & MNIST & Fashion-MNIST & CIFAR-10 & Avg \\")
    latex_lines.append(r"\midrule")
    latex_lines.append(r"Experts (Individual) & - & - & - & 99.14 & 91.65 & 79.91 & 90.23 \\")
    latex_lines.append(r"Weight Averaging (WA) & - & - & - & 49.06 & 43.33 & 20.13 & 37.51 \\")
    latex_lines.append(r"\midrule")
    
    for N in [16, 64, 256]:
        latex_lines.append(f"\\multicolumn{{8}}{{c}}{{\\textbf{{Calibration Budget $N = {N}$}}}} \\\\")
        latex_lines.append(r"\midrule")
        
        # Configurations to display
        configs = [
            ('WA', 'L2-LSHA'),
            ('WA', 'PR-LSHA'),
            ('WFC', 'L2-LSHA'),
            ('WFC', 'PR-LSHA') # Proposed
        ]
        
        for backbone, head in configs:
            entry = best_configs[(N, backbone, head)]
            name_mapping = {
                'WA': 'WA (Direct)',
                'WFC': 'WFC (Calibrated)'
            }
            head_mapping = {
                'L2-LSHA': 'L2-LSHA (Standard)',
                'PR-LSHA': '\\textbf{PR-LSHA (Ours)}' if backbone == 'WFC' else 'PR-LSHA (Prior-Reg)'
            }
            backbone_name = name_mapping[backbone]
            head_name = head_mapping[head]
            
            # Format numbers to .2f
            m_acc = entry['mnist'] * 100
            f_acc = entry['fashion'] * 100
            c_acc = entry['cifar'] * 100
            avg_acc = entry['avg'] * 100
            
            reg_str = f"{entry['reg']:.1f}" if entry['reg'] > 0 else "0.0"
            
            line = f"{backbone_name} & {head_name} & {N} & {reg_str} & {m_acc:.2f} & {f_acc:.2f} & {c_acc:.2f} & \\textbf{{{avg_acc:.2f}}}" if (backbone=='WFC' and head=='PR-LSHA') else f"{backbone_name} & {head_name} & {N} & {reg_str} & {m_acc:.2f} & {f_acc:.2f} & {c_acc:.2f} & {avg_acc:.2f}"
            line += " \\\\"
            latex_lines.append(line)
        latex_lines.append(r"\midrule")
        
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{sc}")
    latex_lines.append(r"\end{small}")
    latex_lines.append(r"\end{center}")
    latex_lines.append(r"\vskip -0.1in")
    latex_lines.append(r"\end{table*}")
    
    latex_code = "\n".join(latex_lines)
    return latex_code

if __name__ == '__main__':
    try:
        with open('ablation_results.json', 'r') as f:
            results = json.load(f)
        
        latex_tbl = format_latex_table(results)
        print("\n=== GENERATED LATEX TABLE ===")
        print(latex_tbl)
        
        with open('ablation_latex_table.tex', 'w') as f:
            f.write(latex_tbl)
        print("\nLaTeX table saved to ablation_latex_table.tex!")
        
    except FileNotFoundError:
        print("ablation_results.json not found. Please wait for the Slurm job to complete.")
