import pandas as pd

def make_table(df, mode, coeff=None):
    # Filter by mode and optionally coeff
    if coeff is not None:
        sub = df[(df["merge_mode"] == mode) & (df["coeff"] == coeff)]
        title = f"Task Arithmetic (TA, $\\lambda={coeff}$)"
        label = f"tab:appendix_ta_{str(coeff).replace('.', '')}"
    else:
        sub = df[df["merge_mode"] == mode]
        title = f"Weight Averaging (WA, $\\lambda=0.3$)"
        label = "tab:appendix_wa_03"
        
    setups = [
        "baseline", "head_sft", "head_tta",
        "lsc", "lsc_head_sft", "lsc_head_tta",
        "n_taac", "n_taac_head_sft", "n_taac_head_tta"
    ]
    
    setup_names = {
        "baseline": "Baseline (No Cal/Adapt)",
        "head_sft": "Head SFT alone",
        "head_tta": "Head TTA alone",
        "lsc": "LSC Calibration",
        "lsc_head_sft": "LSC + Head SFT",
        "lsc_head_tta": "LSC + Head TTA",
        "n_taac": "N-TAAC Calibration",
        "n_taac_head_sft": "\\textbf{N-TAAC + SFT (Ours)}",
        "n_taac_head_tta": "\\textbf{N-TAAC + TTA (Ours)}"
    }
    
    budgets = [4, 8, 16, 32, 64, 128, 256]
    
    lines = []
    lines.append("% " + title)
    lines.append("\\begin{table*}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Comprehensive multi-task average accuracy (\\%) for {title} across varying calibration budgets $N$ per task. Results are averaged over 3 random seeds (standard deviation in parentheses).}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\vskip 0.15in")
    lines.append("\\begin{sc}")
    lines.append("\\begin{scriptsize}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l" + "c"*len(budgets) + "}")
    lines.append("\\toprule")
    header = "Setup & " + " & ".join([f"$N={b}$" for b in budgets]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    for setup in setups:
        setup_row = [setup_names[setup]]
        for b in budgets:
            cell = sub[(sub["setup"] == setup) & (sub["cal_size"] == b)]
            if cell.empty:
                setup_row.append("-")
            else:
                mean = cell["avg_mean"].values[0]
                std = cell["avg_std"].values[0]
                setup_row.append(f"{mean:.2f} ({std:.2f})")
        lines.append(" & ".join(setup_row) + " \\\\")
        
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{scriptsize}")
    lines.append("\\end{sc}")
    lines.append("\\end{table*}")
    lines.append("\n")
    return "\n".join(lines)

def main():
    df = pd.read_csv("collated_aggregated_results.csv")
    
    # Generate tables for WA 0.3, TA 0.1, TA 0.3, TA 0.5, TA 0.7, TA 0.9, TA 1.0
    with open("appendix_tables.tex", "w") as f:
        f.write(make_table(df, "WA"))
        f.write(make_table(df, "TA", 0.1))
        f.write(make_table(df, "TA", 0.3))
        f.write(make_table(df, "TA", 0.5))
        f.write(make_table(df, "TA", 0.7))
        f.write(make_table(df, "TA", 0.9))
        f.write(make_table(df, "TA", 1.0))
        
    print("Appendix tables generated in appendix_tables.tex")

if __name__ == "__main__":
    main()
