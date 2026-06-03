import json
import numpy as np

def process_results():
    seeds = [42, 43, 44]
    results_by_seed = {}
    
    # Check if all files exist
    import os
    missing = []
    for s in seeds:
        path = f"results_cal_sizes_seed{s}.json"
        if not os.path.exists(path):
            missing.append(path)
        else:
            with open(path, "r") as f:
                results_by_seed[s] = json.load(f)
                
    if missing:
        print(f"Error: Missing results files: {missing}")
        return
        
    print("Found all results files. Aggregating results...")
    
    sizes = ["16", "32", "64", "128"]
    scenarios = ["A", "B", "C", "D", "E"]
    methods = ["bnc_avg", "sptaac_avg", "hybrid4_avg", "hybrid8_avg"]
    
    # We want a table for each size, or a combined table where we show the methods and scenarios
    # In submission.tex, the table has rows grouped by size, and columns as Scenarios.
    # Let's see the structure of the existing table:
    # Calibration Size & Method & Scenario A & Scenario B & Scenario C & Scenario D & Scenario E
    # \midrule
    # Uncalibrated & None & ...
    # \midrule
    # N=16 & BNC & ...
    #      & SP-TAAC & ...
    #      & Hybrid (Rank 4) & ...
    #      & Hybrid (Rank 8) & ...
    # and so on.
    
    # First, let's get the uncalibrated numbers (which don't depend on N, but are reported for reference)
    # Let's collect uncal_avg across all seeds for each scenario
    uncal_by_sc = {sc: [] for sc in scenarios}
    for s in seeds:
        for sc in scenarios:
            # We can take it from any size, say "16"
            val = results_by_seed[s]["16"][sc]["uncal_avg"]
            uncal_by_sc[sc].append(val)
            
    uncal_stats = {}
    for sc in scenarios:
        arr = np.array(uncal_by_sc[sc])
        uncal_stats[sc] = (np.mean(arr), np.std(arr))
        
    print("\n--- Uncalibrated Stats (Mean +- SD) ---")
    for sc in scenarios:
        m, sd = uncal_stats[sc]
        print(f"Scenario {sc}: {m:.2f}% +- {sd:.2f}%")
        
    # Now, for each size and method, collect stats across seeds
    stats = {size: {method: {sc: (0.0, 0.0) for sc in scenarios} for method in methods} for size in sizes}
    
    for size in sizes:
        for method in methods:
            for sc in scenarios:
                vals = []
                for s in seeds:
                    val = results_by_seed[s][size][sc][method]
                    vals.append(val)
                arr = np.array(vals)
                stats[size][method][sc] = (np.mean(arr), np.std(arr))
                
    # Now let's print the LaTeX table rows
    print("\n--- Generated LaTeX Table Rows ---")
    
    # Header
    print(r"\begin{table*}[h]")
    print(r"\caption{Effect of Calibration Sample Size ($N$) on Merged Model Test Accuracy (Mean $\pm$ SD across 3 random seeds). We report average test accuracy across the three tasks for different calibration sizes ($N \in \{16, 32, 64, 128\}$) across all five scenarios.}")
    print(r"\label{tab:cal_sizes}")
    print(r"\vskip 0.15in")
    print(r"\begin{center}")
    print(r"\small")
    print(r"\scshape")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Calibration Size & Method & Scenario A & Scenario B & Scenario C & Scenario D & Scenario E \\")
    print(r" & & (SGD Std) & (SGD High Decay) & (AdamW Std) & (AdamW High LR) & (AdamW High Decay) \\")
    print(r"\midrule")
    
    # Print Uncalibrated
    uncal_row = "Uncalibrated & None"
    for sc in scenarios:
        m, sd = uncal_stats[sc]
        uncal_row += f" & {m:.2f}\\% $\\pm$ {sd:.2f}\\%"
    uncal_row += r" \\"
    print(uncal_row)
    print(r"\midrule")
    
    method_name_map = {
        "bnc_avg": "BNC",
        "sptaac_avg": "SP-TAAC",
        "hybrid4_avg": "Hybrid (Rank 4)",
        "hybrid8_avg": "Hybrid (Rank 8)"
    }
    
    for i, size in enumerate(sizes):
        for j, method in enumerate(methods):
            if j == 0:
                row_start = f"$N={size}$ & {method_name_map[method]}"
            else:
                row_start = f" & {method_name_map[method]}"
                
            row = row_start
            for sc in scenarios:
                m, sd = stats[size][method][sc]
                row += f" & {m:.2f}\\% $\\pm$ {sd:.2f}\\%"
            row += r" \\"
            print(row)
        if i < len(sizes) - 1:
            print(r"\midrule")
            
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{center}")
    print(r"\vskip -0.1in")
    print(r"\end{table*}")

if __name__ == "__main__":
    process_results()
