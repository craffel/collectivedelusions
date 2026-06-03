import os
import pandas as pd

def format_table():
    csv_path = "merging_robustness_results.csv"
    tex_path = "submission.tex"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist yet. Please wait for experiments to finish.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Compute the average over tasks (MNIST, Fashion, CIFAR-10) for each group
    summary = df.groupby(["Merge", "N", "Corruption", "Severity"])[["Uncalibrated", "TCAC", "SP-TAAC", "Joint-BN-Adapt", "TC-BN-Adapt"]].mean().reset_index()
    
    # Sort for consistent display: Merge, N, Corruption (order: gaussian, salt_pepper, blur), Severity
    corr_order = {"gaussian": 0, "salt_pepper": 1, "blur": 2}
    summary["corr_idx"] = summary["Corruption"].map(corr_order)
    summary = summary.sort_values(by=["Merge", "N", "corr_idx", "Severity"])
    
    latex_rows = []
    
    prev_merge = None
    prev_n = None
    prev_corr = None
    
    for idx, row in summary.iterrows():
        merge = row["Merge"].upper()
        n = int(row["N"])
        corr = row["Corruption"]
        sev = row["Severity"]
        
        # Map corruption names to LaTeX readable text
        corr_map = {
            "gaussian": "Gaussian",
            "salt_pepper": "Salt \\& Pepper",
            "blur": "Blur"
        }
        corr_str_val = corr_map.get(corr, corr.capitalize())
        
        uncal = f"{row['Uncalibrated']:.2f}\%"
        tcac = f"{row['TCAC']:.2f}\%"
        sp_taac = f"{row['SP-TAAC']:.2f}\%"
        joint_bn = f"{row['Joint-BN-Adapt']:.2f}\%"
        tc_bn = f"{row['TC-BN-Adapt']:.2f}\%"
        
        # Highlight the best performing method in bold (among the calibrated/adapted ones)
        methods = [row['TCAC'], row['SP-TAAC'], row['Joint-BN-Adapt'], row['TC-BN-Adapt']]
        max_val = max(methods)
        
        if row['TCAC'] == max_val:
            tcac = f"\\textbf{{{tcac}}}"
        if row['SP-TAAC'] == max_val:
            sp_taac = f"\\textbf{{{sp_taac}}}"
        if row['Joint-BN-Adapt'] == max_val:
            joint_bn = f"\\textbf{{{joint_bn}}}"
        if row['TC-BN-Adapt'] == max_val:
            tc_bn = f"\\textbf{{{tc_bn}}}"
            
        merge_str = merge if merge != prev_merge else ""
        n_str = str(n) if (n != prev_n or merge != prev_merge) else ""
        corr_str = corr_str_val if (corr != prev_corr or n != prev_n or merge != prev_merge) else ""
        
        latex_rows.append(f"{merge_str} & {n_str} & {corr_str} & {sev:.1f} & {uncal} & {tcac} & {sp_taac} & {joint_bn} & {tc_bn} \\\\")
        
        # Add spacing/midrule between groups
        if prev_n is not None and n != prev_n and merge == prev_merge:
            latex_rows.insert(-1, "\\midrule")
        elif prev_merge is not None and merge != prev_merge:
            latex_rows.insert(-1, "\\midrule\\midrule")
            
        prev_merge = merge
        prev_n = n
        prev_corr = corr
        
    latex_table_content = "\n".join(latex_rows)
    
    # Read submission.tex
    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()
        
    # We find where to inject: we look for the tabular definition with 9 columns or the old 8 columns, and the midrule that follows.
    # To be extremely safe, let's search for \begin{tabular}{...} and the headers, and replace the rows.
    # Specifically, we want to locate the tabular block:
    # \begin{tabular}{lcccccccc}
    # \toprule
    # \textbf{Merge} & \textbf{N} & \textbf{Corruption} & \textbf{Severity} & ...
    # \midrule
    # <rows>
    # \bottomrule
    # \end{tabular}
    
    import re
    
    # Find the tabular section
    pattern = r"(\\begin\{tabular\}\{[lc]+\}\s*\\toprule\s*\\textbf\{Merge\}.*?\\midrule)(.*?)(\\bottomrule\s*\\end\{tabular\})"
    
    match = re.search(pattern, tex_content, re.DOTALL)
    if match:
        header_part = match.group(1)
        footer_part = match.group(3)
        
        # Let's check if the header has 9 columns. If not, let's replace the header part to have 9 columns
        if "Corruption" not in header_part:
            header_part = "\\begin{tabular}{lcccccccc}\n\\toprule\n\\textbf{Merge} & \\textbf{N} & \\textbf{Corruption} & \\textbf{Severity} & \\textbf{Uncalibrated} & \\textbf{TCAC} & \\textbf{SP-TAAC} & \\textbf{Joint-BN-Adapt} & \\textbf{TC-BN-Adapt} \\\\\n\\midrule"
            
        new_tabular = f"{header_part}\n{latex_table_content}\n{footer_part}"
        tex_content = tex_content[:match.start()] + new_tabular + tex_content[match.end():]
        
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
        print("Successfully injected results table into submission.tex!")
    else:
        print("Error: Could not locate tabular block in submission.tex via regex!")

if __name__ == "__main__":
    format_table()
