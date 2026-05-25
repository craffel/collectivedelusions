import glob
import re
import os

def parse_output_file(file_pattern):
    files = glob.glob(file_pattern)
    if not files:
        return None
    
    # Sort by modification time to get the latest if there are duplicates
    files.sort(key=os.path.getmtime)
    latest_file = files[-1]
    
    results = {}
    print(f"Parsing {latest_file}...")
    with open(latest_file, "r") as f:
        content = f.read()
        
    # Find table lines
    lines = content.split("\n")
    table_started = False
    for line in lines:
        if "Algorithm" in line and "Sequential" in line:
            table_started = True
            continue
        if table_started:
            if "===" in line or "---" in line or line.strip() == "":
                if len(results) > 0:
                    # End of table
                    break
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                algo_name = parts[0]
                try:
                    seq_acc = float(parts[1])
                    alt_acc = float(parts[2])
                    results[algo_name] = {"seq": seq_acc, "alt": alt_acc}
                except ValueError:
                    pass
    return results

def main():
    job_patterns = {
        "base": "ttmm-base_*.out",
        "floor_0.01": "ttmm-floor-0.01_*.out",
        "floor_0.05": "ttmm-floor-0.05_*.out",
        "floor_0.1": "ttmm-floor-0.1_*.out",
        "steps_3": "ttmm-floor-0.05-steps-3_*.out",
        "steps_5": "ttmm-floor-0.05-steps-5_*.out",
        "beta_0.0": "ttmm-beta-0.0_*.out",
        "beta_0.5": "ttmm-beta-0.5_*.out",
        "beta_0.95": "ttmm-beta-0.95_*.out",
        "beta_0.99": "ttmm-beta-0.99_*.out"
    }
    
    all_data = {}
    for key, pattern in job_patterns.items():
        res = parse_output_file(pattern)
        if res:
            all_data[key] = res
        else:
            print(f"Warning: No files found matching pattern {pattern}")
            
    if len(all_data) < len(job_patterns):
        print("Some jobs are still running or failed. Please run this script again once all jobs complete.")
        return
        
    print("\n--- Ablation Results Parsed Successfully ---")
    
    # Generate LaTeX table
    latex_table = r"""\begin{table}[t]
\caption{Ablation studies on the sensitivity floor $\tau$, test-time adaptation steps per batch $S$, and momentum decay factor $\beta$ on the Sequential and Alternating streams. Bold indicates best performance within each study block.}
\label{tab:ablation_results}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccccc}
\toprule
Algorithm & Floor $\tau$ & Steps $S$ & Momentum $\beta$ & Sequential Acc (\%) & Alternating Acc (\%) \\
\midrule
"""
    
    # Block 1: FP-CA vs. FP-CA + TMS (Base, no floor, 1 step)
    b_fpca_seq = all_data["base"]["FP-CA"]["seq"]
    b_fpca_alt = all_data["base"]["FP-CA"]["alt"]
    b_fpca_tms_seq = all_data["base"]["FP-CA + TMS (Ours)"]["seq"]
    b_fpca_tms_alt = all_data["base"]["FP-CA + TMS (Ours)"]["alt"]
    
    b_iggs_seq = all_data["base"]["IGGS-Merge"]["seq"]
    b_iggs_alt = all_data["base"]["IGGS-Merge"]["alt"]
    b_iggs_tms_seq = all_data["base"]["IGGS-Merge + TMS (Ours)"]["seq"]
    b_iggs_tms_alt = all_data["base"]["IGGS-Merge + TMS (Ours)"]["alt"]
    
    latex_table += f"FP-CA & $10^{{-6}}$ & 1 & 0.9 & {b_fpca_seq:.2f} & {b_fpca_alt:.2f} \\\\\n"
    latex_table += f"FP-CA + TMS & $10^{{-6}}$ & 1 & 0.9 & {b_fpca_tms_seq:.2f} & {b_fpca_tms_alt:.2f} \\\\\n"
    latex_table += f"IGGS-Merge & $10^{{-6}}$ & 1 & 0.9 & {b_iggs_seq:.2f} & {b_iggs_alt:.2f} \\\\\n"
    latex_table += f"IGGS-Merge + TMS & $10^{{-6}}$ & 1 & 0.9 & {b_iggs_tms_seq:.2f} & {b_iggs_tms_alt:.2f} \\\\\n"
    latex_table += "\\midrule\n"
    
    # Block 2: Sensitivity Floor Sweep with 1 step
    # We compare FP-CA + TMS and IGGS-Merge + TMS under floors 0.01, 0.05, 0.1
    for floor_val, floor_key in [("0.01", "floor_0.01"), ("0.05", "floor_0.05"), ("0.1", "floor_0.1")]:
        fp_seq = all_data[floor_key]["FP-CA + TMS (Ours)"]["seq"]
        fp_alt = all_data[floor_key]["FP-CA + TMS (Ours)"]["alt"]
        ig_seq = all_data[floor_key]["IGGS-Merge + TMS (Ours)"]["seq"]
        ig_alt = all_data[floor_key]["IGGS-Merge + TMS (Ours)"]["alt"]
        
        latex_table += f"FP-CA + TMS & {floor_val} & 1 & 0.9 & {fp_seq:.2f} & {fp_alt:.2f} \\\\\n"
        latex_table += f"IGGS-Merge + TMS & {floor_val} & 1 & 0.9 & {ig_seq:.2f} & {ig_alt:.2f} \\\\\n"
        
    latex_table += "\\midrule\n"
    
    # Block 3: Adaptation Steps Sweep with floor 0.05
    for steps_val, steps_key in [("3", "steps_3"), ("5", "steps_5")]:
        fp_seq = all_data[steps_key]["FP-CA + TMS (Ours)"]["seq"]
        fp_alt = all_data[steps_key]["FP-CA + TMS (Ours)"]["alt"]
        ig_seq = all_data[steps_key]["IGGS-Merge + TMS (Ours)"]["seq"]
        ig_alt = all_data[steps_key]["IGGS-Merge + TMS (Ours)"]["alt"]
        
        latex_table += f"FP-CA + TMS & 0.05 & {steps_val} & 0.9 & {fp_seq:.2f} & {fp_alt:.2f} \\\\\n"
        latex_table += f"IGGS-Merge + TMS & 0.05 & {steps_val} & 0.9 & {ig_seq:.2f} & {ig_alt:.2f} \\\\\n"
        
    latex_table += "\\midrule\n"

    # Block 4: Momentum Decay Sweep with floor 0.01 and 1 step
    for beta_val, beta_key in [("0.0", "beta_0.0"), ("0.5", "beta_0.5"), ("0.9", "floor_0.01"), ("0.95", "beta_0.95"), ("0.99", "beta_0.99")]:
        fp_seq = all_data[beta_key]["FP-CA + TMS (Ours)"]["seq"]
        fp_alt = all_data[beta_key]["FP-CA + TMS (Ours)"]["alt"]
        ig_seq = all_data[beta_key]["IGGS-Merge + TMS (Ours)"]["seq"]
        ig_alt = all_data[beta_key]["IGGS-Merge + TMS (Ours)"]["alt"]
        
        latex_table += f"FP-CA + TMS & 0.01 & 1 & {beta_val} & {fp_seq:.2f} & {fp_alt:.2f} \\\\\n"
        latex_table += f"IGGS-Merge + TMS & 0.01 & 1 & {beta_val} & {ig_seq:.2f} & {ig_alt:.2f} \\\\\n"
        
    latex_table += r"""\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}"""

    # Insert this table into template/example_paper.tex
    paper_path = "template/example_paper.tex"
    print(f"Reading {paper_path}...")
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace the existing table containing the ablation results label if present
    label_pos = content.find("\\label{tab:ablation_results}")
    if label_pos != -1:
        # Find the nearest \begin{table} before the label
        start_pos = content.rfind("\\begin{table}", 0, label_pos)
        # Find the nearest \end{table} after the label
        end_pos = content.find("\\end{table}", label_pos)
        if start_pos != -1 and end_pos != -1:
            end_pos += len("\\end{table}")
            print("Found existing Ablation Studies Table. Replacing it...")
            new_content = content[:start_pos] + latex_table + content[end_pos:]
            with open(paper_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("Successfully updated paper with the Ablation Studies Table!")
        else:
            print("Warning: Could not find table boundaries around label.")
    else:
        print("Warning: Label 'tab:ablation_results' not found.")
        
    # Run the compilation script to rebuild submission.pdf
    print("Rebuilding paper PDF...")
    try:
        import subprocess
        result = subprocess.run(["python3", "process_results_and_compile.py"], capture_output=True, text=True, check=True)
        print("Paper successfully recompiled to submission.pdf!")
    except subprocess.CalledProcessError as e:
        print("Recompilation failed with error:")
        print(e.stderr)
        print(e.stdout)

if __name__ == "__main__":
    main()
