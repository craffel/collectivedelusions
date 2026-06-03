import json
import numpy as np

def main():
    print("Loading results from checkpoints/experimental_results.json...")
    with open("checkpoints/experimental_results.json", "r") as f:
        data = json.load(f)
        
    results = data["main_results"]
    ablation_results = data.get("ablation_results", {})
    
    merge_modes = ["WA", "TA"]
    calibration_sizes = [8, 16, 32, 64, 128]
    methods = ["uncalibrated", "ntaac", "tcbna", "sptaac", "csc", "wrsc"]
    
    method_names = {
        "uncalibrated": "Uncalibrated Baseline",
        "ntaac": "N-TAAC (Agnostic BN)",
        "tcbna": "TC-BNA (Conditional BN)",
        "sptaac": "SP-TAAC (Global Scaling)",
        "csc": "CSC (Channel-wise)",
        "wrsc": "WRSC (Ours, $\\alpha=0.05$)"
    }
    
    # We want to format each entry as "mean \pm std"
    # And we bold the highest value in each column per merge mode!
    means = {mode: {m: {} for m in methods} for mode in merge_modes}
    stds = {mode: {m: {} for m in methods} for mode in merge_modes}
    
    for mode in merge_modes:
        for m in methods:
            for size in calibration_sizes:
                size_str = str(size)
                seed_res = results[mode][m].get(size_str, [])
                if not seed_res:
                    seed_res = results[mode][m].get(size, [])
                
                if seed_res:
                    averages = [s["average"] for s in seed_res]
                    mean_val = np.mean(averages)
                    std_val = np.std(averages)
                else:
                    mean_val, std_val = 0.0, 0.0
                    
                means[mode][m][size] = mean_val
                stds[mode][m][size] = std_val
                
    # Find max mean in each column per merge mode to bold it
    max_means = {mode: {size: 0.0 for size in calibration_sizes} for mode in merge_modes}
    for mode in merge_modes:
        for size in calibration_sizes:
            col_means = [means[mode][m][size] for m in methods]
            max_means[mode][size] = max(col_means) if col_means else 0.0
            
    # Generate LaTeX table code
    latex_lines = []
    latex_lines.append("\\begin{tabular}{llccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Merge Mode & Calibration Method & $N=8$ & $N=16$ & $N=32$ & $N=64$ & $N=128$ \\\\")
    latex_lines.append("\\midrule")
    
    for mode in merge_modes:
        for m in methods:
            row_vals = []
            for size in calibration_sizes:
                mean_val = means[mode][m][size]
                std_val = stds[mode][m][size]
                
                is_max = (abs(mean_val - max_means[mode][size]) < 1e-5)
                
                if is_max:
                    entry = f"\\textbf{{{mean_val:.2f} $\\pm$ {std_val:.2f}}}"
                else:
                    entry = f"{mean_val:.2f} $\\pm$ {std_val:.2f}"
                row_vals.append(entry)
                
            method_disp = method_names[m]
            row_str = f"{mode} & {method_disp} & " + " & ".join(row_vals) + " \\\\"
            latex_lines.append(row_str)
            
        if mode == "WA":
            latex_lines.append("\\midrule")
            
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    
    latex_table_code = "\n".join(latex_lines)
    
    # Read the LaTeX template file
    print("Reading template/example_paper.tex...")
    with open("template/example_paper.tex", "r") as f:
        paper_content = f.read()
        
    # Replace using slicing for robust literal replacement without regex backslash issues
    start_idx = paper_content.find("\\begin{tabular}{llccc}")
    if start_idx == -1:
        start_idx = paper_content.find("\\begin{tabular}{llccccc}")
        
    if start_idx != -1:
        end_idx = paper_content.find("\\end{tabular}", start_idx)
        if end_idx != -1:
            end_idx += len("\\end{tabular}")
            new_paper_content = paper_content[:start_idx] + latex_table_code + paper_content[end_idx:]
            print("Successfully replaced tabular block using index slicing.")
            
            with open("template/example_paper.tex", "w") as f:
                f.write(new_paper_content)
            print("Updated template/example_paper.tex successfully.")
        else:
            print("ERROR: Found begin{tabular} but could not find matching end{tabular}!")
    else:
        print("ERROR: Could not find begin{tabular}{llccc} or begin{tabular}{llccccc} in template/example_paper.tex!")

if __name__ == "__main__":
    main()
