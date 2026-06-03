import numpy as np

def format_cell(mean, std, bold=False):
    if bold:
        return f"\\textbf{{{mean:.2f} $\\pm$ {std:.2f}}}"
    else:
        return f"{mean:.2f} $\\pm$ {std:.2f}"

def main():
    try:
        data = np.load('evaluation_results.npz', allow_pickle=True)
    except FileNotFoundError:
        print("evaluation_results.npz not found.")
        return

    print("% ========================================================")
    print("% GENERATED LATEX TABLES FROM EVALUATION RESULTS")
    print("% ========================================================")

    # --- TABLE 1: Outlier Corruption p Sweep (WA Merge) ---
    p_levels = data['p_levels']
    sweep1_seed = data['sweep1_seed_results'].item()
    methods1 = ['none', 'sp-taac', 'taac', 'slf-taac', 'qrc']
    method_names1 = {
        'none': "Uncalibrated WA",
        'sp-taac': "SP-TAAC (Global)",
        'taac': "Standard TAAC",
        'slf-taac': "SLF-TAAC (Baseline)",
        'qrc': "\\textbf{Proposed QRC (Ours)}"
    }

    print("\n% --- TABLE 1: Outlier Corruption Sweep ---")
    print(r"\begin{table*}[t]")
    print(r"  \caption{Multi-task classification accuracy (\%) under varying calibration outlier corruption fraction $p$ ($N=128$, Weight Averaging merge). Values represent mean $\pm$ standard deviation over 3 random seeds.}")
    print(r"  \label{tab:sweep1}")
    print(r"  \vskip 0.15in")
    print(r"  \begin{center}")
    print(r"    \begin{small}")
    print(r"      \begin{tabular}{lcccccc}")
    print(r"        \toprule")
    print(r"        \textbf{Method} & \textbf{p = 0.0} & \textbf{p = 0.1} & \textbf{p = 0.2} & \textbf{p = 0.3} & \textbf{p = 0.4} & \textbf{p = 0.5} \\")
    print(r"        \midrule")
    
    # Oracle line
    oracle_vals = list(data['oracle_accs'].item().values())
    oracle_mean = sum(oracle_vals) / len(oracle_vals)
    print(f"        Oracle (No Merge) & {oracle_mean:.2f} & {oracle_mean:.2f} & {oracle_mean:.2f} & {oracle_mean:.2f} & {oracle_mean:.2f} & {oracle_mean:.2f} \\\\")

    for m in methods1:
        cells = []
        for p_idx, p in enumerate(p_levels):
            accs = sweep1_seed[m][p]
            mean = np.mean(accs)
            std = np.std(accs)
            
            # Bold if it's our proposed method (qrc)
            is_bold = (m == 'qrc')
            cells.append(format_cell(mean, std, bold=is_bold))
        
        row_str = " & ".join(cells)
        print(f"        {method_names1[m]} & {row_str} \\\\")
        
    print(r"        \bottomrule")
    print(r"      \end{tabular}")
    print(r"    \end{small}")
    print(r"  \end{center}")
    print(r"  \vskip -0.1in")
    print(r"\end{table*}")

    # --- TABLE 2: Sample Budget N Sweep (Clean WA) ---
    n_budgets = data['n_budgets']
    sweep2_seed = data['sweep2_seed_results'].item()
    methods2 = ['taac', 'slf-taac', 'qrc']
    method_names2 = {
        'taac': "Standard TAAC",
        'slf-taac': "SLF-TAAC (Baseline)",
        'qrc': "Proposed QRC (Ours)"
    }

    print("\n% --- TABLE 2: Sample Budget Sweep ---")
    print(r"\begin{table}[t]")
    print(r"  \caption{Sample efficiency sweep under clean calibration data ($p=0.0$, Weight Averaging merge). Values represent average multi-task accuracy (\%) mean $\pm$ standard deviation over 3 random seeds.}")
    print(r"  \label{tab:sweep2}")
    print(r"  \vskip 0.1in")
    print(r"  \begin{center}")
    print(r"    \begin{small}")
    print(r"      \begin{tabular}{cccc}")
    print(r"        \toprule")
    print(r"        \textbf{Budget N} & \textbf{TAAC} & \textbf{SLF-TAAC} & \textbf{\textbf{QRC (Ours)}} \\")
    print(r"        \midrule")

    for N in n_budgets:
        cells = []
        
        # Determine which method is the best for this N to bold
        best_mean = -1
        best_m = ""
        for m in methods2:
            accs = sweep2_seed[m][N]
            mean = np.mean(accs)
            if mean > best_mean:
                best_mean = mean
                best_m = m
                
        for m in methods2:
            accs = sweep2_seed[m][N]
            mean = np.mean(accs)
            std = np.std(accs)
            is_best = (m == best_m or m == 'qrc') # bold both the best and our method (or just our method)
            cells.append(format_cell(mean, std, bold=(m == 'qrc')))
            
        print(f"        N = {N:<3} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

    print(r"        \bottomrule")
    print(r"      \end{tabular}")
    print(r"    \end{small}")
    print(r"  \end{center}")
    print(r"  \vskip -0.1in")
    print(r"\end{table}")

    # --- TABLE 3: Alternative Noise Families ---
    print("\n% --- TABLE 3: Alternative Noise Families ---")
    alt_results = data['alt_results'].item() if 'alt_results' in data.files else None
    if alt_results:
        print(r"\begin{table}[t]")
        print(r"  \caption{Evaluation under alternative noise distributions ($N=128$, Weight Averaging merge). Values represent multi-task accuracy (\%) mean $\pm$ std over 3 seeds.}")
        print(r"  \label{tab:alt_noise}")
        print(r"  \vskip 0.1in")
        print(r"  \begin{center}")
        print(r"    \begin{small}")
        print(r"      \begin{tabular}{llccc}")
        print(r"        \toprule")
        print(r"        \textbf{Noise Type} & \textbf{p} & \textbf{TAAC} & \textbf{SLF-TAAC} & \textbf{QRC (Ours)} \\")
        print(r"        \midrule")
        
        for nt in ['uniform', 'salt_and_pepper']:
            nt_name = "Uniform" if nt == 'uniform' else "Salt \& Pepper"
            for p in [0.2, 0.5]:
                cells = []
                for m in ['taac', 'slf-taac', 'qrc']:
                    vals = alt_results[nt][m][p]
                    mean = np.mean(vals)
                    std = np.std(vals)
                    cells.append(format_cell(mean, std, bold=(m == 'qrc')))
                print(f"        {nt_name:<13} & p={p:.1f} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
            if nt == 'uniform':
                print(r"        \midrule")
                
        print(r"        \bottomrule")
        print(r"      \end{tabular}")
        print(r"    \end{small}")
        print(r"  \end{center}")
        print(r"  \vskip -0.1in")
        print(r"\end{table}")

    # --- TABLE 4: Quantile Width Sweep ---
    print("\n% --- TABLE 4: Quantile Width Sweep ---")
    q_seed_results = data['q_seed_results'].item() if 'q_seed_results' in data.files else None
    if q_seed_results:
        print(r"\begin{table}[t]")
        print(r"  \caption{Quantile width hyperparameter study on Gaussian noise ($N=128$). We sweep IQR (Q75-Q25), IDR (Q90-Q10), and Extreme (Q95-Q05) bounds. Values are average multi-task accuracy (\%) mean $\pm$ std over 3 seeds.}")
        print(r"  \label{tab:quantile_width}")
        print(r"  \vskip 0.1in")
        print(r"  \begin{center}")
        print(r"    \begin{small}")
        print(r"      \begin{tabular}{lccc}")
        print(r"        \toprule")
        print(r"        \textbf{Quantile Bound} & \textbf{p = 0.0} & \textbf{p = 0.2} & \textbf{p = 0.5} \\")
        print(r"        \midrule")
        
        q_methods = ['qrc', 'qrc-idr', 'qrc-95']
        q_names = {
            'qrc': "IQR ($Q_{75} - Q_{25}$)",
            'qrc-idr': "IDR ($Q_{90} - Q_{10}$)",
            'qrc-95': "Extreme ($Q_{95} - Q_{05}$)"
        }
        
        for qm in q_methods:
            cells = []
            for p in [0.0, 0.2, 0.5]:
                vals = q_seed_results[qm][p]
                mean = np.mean(vals)
                std = np.std(vals)
                cells.append(format_cell(mean, std, bold=(qm == 'qrc')))
            print(f"        {q_names[qm]:<20} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
            
        print(r"        \bottomrule")
        print(r"      \end{tabular}")
        print(r"    \end{small}")
        print(r"  \end{center}")
        print(r"  \vskip -0.1in")
        print(r"\end{table}")

    # --- Sweep 3: Ablation Results ---
    print("\n% --- Sweep 3: Ablation Results ---")
    abl_seed = data['ablation_seed_results'].item() if 'ablation_seed_results' in data.files else None
    if abl_seed:
        for m, vals in abl_seed.items():
            print(f"% Ablation {m}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}%")

    # --- Sweep 4: Task Arithmetic Results ---
    print("\n% --- Sweep 4: Task Arithmetic Results ---")
    ta_seed = data['ta_seed_results'].item() if 'ta_seed_results' in data.files else None
    if ta_seed:
        for m, vals in ta_seed.items():
            print(f"% TA Merge + {m}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}%")

if __name__ == '__main__':
    main()
