import re

def update_table_in_latex(tex_content, table_label, row_mappings, results):
    # Find the block around the table label
    label_idx = tex_content.find(f"\\label{{{table_label}}}")
    if label_idx == -1:
        print(f"Warning: Label {table_label} not found in LaTeX!")
        return tex_content
    
    # Search for all tabular starts and ends in the file
    start_match = list(re.finditer(r"\\begin\{tabular\}", tex_content))
    end_match = list(re.finditer(r"\\end\{tabular\}", tex_content))
    
    # Find the tabular closest to the label
    best_tabular_start = -1
    best_tabular_end = -1
    min_dist = float('inf')
    for sm, em in zip(start_match, end_match):
        dist = abs(sm.start() - label_idx)
        if dist < min_dist:
            min_dist = dist
            best_tabular_start = sm.start()
            best_tabular_end = em.end()
            
    if best_tabular_start == -1:
        print(f"Warning: tabular block not found for {table_label}!")
        return tex_content
        
    tabular_content = tex_content[best_tabular_start:best_tabular_end]
    lines = tabular_content.split("\n")
    
    replaced_rows = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        for prefix, md_key in row_mappings.items():
            if stripped.startswith(prefix):
                vals = results[md_key]
                new_line = prefix
                for i in range(4):
                    new_line += f" {vals[i][0]:.2f} $\\pm$ {vals[i][1]:.2f}\\% &"
                new_line += f" \\textbf{{{vals[4][0]:.2f} $\\pm$ {vals[4][1]:.2f}\\%}} \\\\"
                
                indent = line[:len(line) - len(line.lstrip())]
                lines[idx] = indent + new_line
                replaced_rows += 1
                break
                
    new_tabular_content = "\n".join(lines)
    tex_content = tex_content[:best_tabular_start] + new_tabular_content + tex_content[best_tabular_end:]
    print(f"Successfully updated {replaced_rows} rows in table {table_label}.")
    return tex_content

def main():
    # 1. Parse experiment_results.md
    with open("experiment_results.md", "r") as f:
        lines = f.readlines()

    results = {}
    for line in lines:
        line = line.strip()
        if line.startswith("|") and not "MNIST" in line and not "---" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 6:
                name = parts[0]
                values = []
                for v in parts[1:6]:
                    v = v.replace("**", "").replace("%", "").strip()
                    v_parts = re.split(r'±|±', v)
                    mean = float(v_parts[0].strip())
                    std = float(v_parts[1].strip())
                    values.append((mean, std))
                results[name] = values

    # 2. Define row mappings for each table
    
    # Table 1: tab:unquantized_results (in appendix.tex)
    table1_mappings = {
        "Individual Experts (FP16) &": "Individual Experts (FP16)",
        "FP16 Uniform Merged (0.3) &": "FP16 Uniform Merged (0.3)",
        "AdaMerging (FP16 ES) &": "AdaMerging (FP16 ES)",
        "AdaMerging (FP16 Adam) &": "AdaMerging (FP16 Adam)",
        "PolyMerge (FP16 Adam) &": "PolyMerge (FP16 Adam)",
    }

    # Table 2: tab:8bit_results (in 04_experiments.tex)
    table2_mappings = {
        "Individual Experts &": "Individual Experts (8-Bit)",
        "Q-then-M (8-Bit) &": "Q-then-M (8-Bit)",
        "M-then-Q (8-Bit) &": "M-then-Q (8-Bit)",
        "AdaMerging (ES) &": "AdaMerging (FP16 ES -> 8-Bit)",
        "AdaMerging (Adam) &": "AdaMerging (FP16 Adam -> 8-Bit)",
        "Q-Merge (ES) &": "Q-Merge (8-Bit ES)",
        "Q-Merge (Adam STE) &": "Q-Merge (8-Bit Adam STE)",
        "\\textbf{Q-PolyMerge (ES)} &": "Q-PolyMerge (8-Bit ES, Proposed)",
        "\\textbf{Q-PolyMerge (Adam)} &": "Q-PolyMerge (8-Bit Adam STE, Proposed)",
    }

    # Table 3: tab:4bit_results (in 04_experiments.tex)
    table3_mappings = {
        "Individual Experts &": "Individual Experts (4-Bit)",
        "Q-then-M (4-Bit) &": "Q-then-M (4-Bit)",
        "M-then-Q (4-Bit) &": "M-then-Q (4-Bit)",
        "AdaMerging (ES) &": "AdaMerging (FP16 ES -> 4-Bit)",
        "AdaMerging (Adam) &": "AdaMerging (FP16 Adam -> 4-Bit)",
        "Q-Merge (ES) &": "Q-Merge (4-Bit ES)",
        "Q-Merge (Adam STE) &": "Q-Merge (4-Bit Adam STE)",
        "\\textbf{Q-PolyMerge (ES)} &": "Q-PolyMerge (4-Bit ES, Proposed)",
        "\\textbf{Q-PolyMerge (Adam)} &": "Q-PolyMerge (4-Bit Adam STE, Proposed)",
    }

    # Table 4: tab:degree_ablation (in appendix.tex)
    table4_mappings = {
        "Linear ($d=1$) &": "Linear (d=1)",
        "Quadratic ($d=2$, Ours) &": "Quadratic (d=2, Proposed)",
        "Cubic ($d=3$) &": "Cubic (d=3)",
        "Quartic ($d=4$) &": "Quartic (d=4)",
    }

    # Table 5: tab:blockwise_comparison (in appendix.tex)
    table5_mappings = {
        "Block-wise Constant (ES) &": "Block-wise Constant (ES)",
        "Block-wise Constant (Adam STE) &": "Block-wise Constant (Adam STE)",
        "Polynomial Continuous (ES, Ours) &": "Polynomial Continuous (ES, Ours)",
        "Polynomial Continuous (Adam STE, Ours) &": "Polynomial Continuous (Adam STE, Ours)",
    }

    # 3. Update 04_experiments.tex
    tex_path = "submission/sections/04_experiments.tex"
    with open(tex_path, "r") as f:
        tex_content = f.read()

    # Clean markdown bold leak inside 04_experiments.tex
    tex_content = tex_content.replace("**47.8\\% reduction in standard deviation**", "\\textbf{47.8\\% reduction in standard deviation}")

    tex_content = update_table_in_latex(tex_content, "tab:8bit_results", table2_mappings, results)
    tex_content = update_table_in_latex(tex_content, "tab:4bit_results", table3_mappings, results)

    # Parse key narrative numbers
    pm_8_avg_mean = results["Q-PolyMerge (8-Bit Adam STE, Proposed)"][4][0]
    qm_8_avg_mean = results["Q-Merge (8-Bit Adam STE)"][4][0]
    u_8_avg_mean = results["M-then-Q (8-Bit)"][4][0]
    ceiling_avg_mean = results["PolyMerge (FP16 Adam)"][4][0]
    
    pm_4_avg_mean = results["Polynomial Continuous (Adam STE, Ours)"][4][0]
    qm_4_avg_mean = results["Q-Merge (4-Bit Adam STE)"][4][0]
    u_4_avg_mean = results["M-then-Q (4-Bit)"][4][0]
    indiv_4_avg_mean = results["Individual Experts (4-Bit)"][4][0]
    
    pm_8_es_mean = results["Q-PolyMerge (8-Bit ES, Proposed)"][4][0]
    qm_8_es_mean = results["Q-Merge (8-Bit ES)"][4][0]
    am_8_es_mean = results["AdaMerging (FP16 ES -> 8-Bit)"][4][0]
    
    pm_4_es_mean = results["Polynomial Continuous (ES, Ours)"][4][0]
    qm_4_es_mean = results["Q-Merge (4-Bit ES)"][4][0]
    am_4_es_mean = results["AdaMerging (FP16 ES -> 4-Bit)"][4][0]
    
    pm_4_adam_mean = results["Polynomial Continuous (Adam STE, Ours)"][4][0]
    am_4_adam_mean = results["AdaMerging (FP16 Adam -> 4-Bit)"][4][0]

    # Update 8-bit text narrative
    tex_content = re.sub(
        r"naive merging followed by quantization \(M-then-Q\) achieves only \\textbf\{[0-9.]+\\\%\}\s+average accuracy",
        lambda m: f"naive merging followed by quantization (M-then-Q) achieves only \\textbf{{{u_8_avg_mean:.2f}\\%}} average accuracy",
        tex_content
    )
    tex_content = re.sub(
        r"optimization over first-order gradient descent \(\\texttt\{Q-Merge\} Adam STE\), the performance reaches \\textbf\{[0-9.]+\\\%\}\.",
        lambda m: f"optimization over first-order gradient descent (\\texttt{{Q-Merge}} Adam STE), the performance reaches \\textbf{{{qm_8_avg_mean:.2f}\\%}}.",
        tex_content
    )
    tex_content = re.sub(
        r"Q-PolyMerge achieves an average accuracy of \\textbf\{[0-9.]+\\\%\}, which is highly comparable to the unconstrained Q-Merge baseline \(\\textbf\{[0-9.]+\\\%\}\) and nearly recovers the unquantized continuous PolyMerge performance ceiling \(\\textbf\{[0-9.]+\\\%\}\s+of FP16 PolyMerge\)\.",
        lambda m: f"Q-PolyMerge achieves an average accuracy of \\textbf{{{pm_8_avg_mean:.2f}\\%}}, which is highly comparable to the unconstrained Q-Merge baseline (\\textbf{{{qm_8_avg_mean:.2f}\\%}}) and nearly recovers the unquantized continuous PolyMerge performance ceiling (\\textbf{{{ceiling_avg_mean:.2f}\\%}} of FP16 PolyMerge).",
        tex_content
    )
    tex_content = re.sub(
        r"\\texttt\{AdaMerging \(ES \\to 8-Bit\)\} achieves only \\textbf\{[0-9.]+\\\%\}\s+average accuracy\..*proposed \\texttt\{Q-PolyMerge \(ES\)\} successfully filters out high-dimensional search noise, achieving \\textbf\{[0-9.]+\\\%\}\s+average accuracy, representing a substantial \\textbf\{([+-][0-9.]+)\\\%\}\s+absolute improvement\.",
        lambda m: f"\\texttt{{AdaMerging (ES \\to 8-Bit)}} achieves only \\textbf{{{am_8_es_mean:.2f}\\%}} average accuracy. In contrast, our proposed \\texttt{{Q-PolyMerge (ES)}} successfully filters out high-dimensional search noise, achieving \\textbf{{{pm_8_es_mean:.2f}\\%}} average accuracy (a substantial \\textbf{{+{pm_8_es_mean - am_8_es_mean:.2f}\\%}} absolute improvement).",
        tex_content,
        flags=re.DOTALL
    )

    # Update 4-bit text narrative
    tex_content = re.sub(
        r"individual experts achieve a healthy average accuracy of \\textbf\{[0-9.]+\\\%\}, but naive merging followed by quantization \(M-then-Q\) drops to \\textbf\{[0-9.]+\\\%\}\s+due to severe representation misalignment",
        lambda m: f"individual experts achieve a healthy average accuracy of \\textbf{{{indiv_4_avg_mean:.2f}\\%}}, but naive merging followed by quantization (M-then-Q) drops to \\textbf{{{u_4_avg_mean:.2f}\\%}} due to severe representation misalignment",
        tex_content
    )
    tex_content = re.sub(
        r"While unconstrained \\texttt\{Q-Merge \(Adam STE\)\} recovers some performance \(\\textbf\{[0-9.]+\\\%\}\), it remains limited by high-dimensional overfitting.*Our proposed \\textbf\{Q-PolyMerge \(Adam STE\)\} achieves the highest performance at \\textbf\{[0-9.]+\\\%\}, representing a significant \\textbf\{([+-][0-9.]+)\\\%\}\s+absolute improvement over the unconstrained baseline and a \\textbf\{([+-][0-9.]+)\\\%\}\s+absolute improvement over naive post-merge quantization\.",
        lambda m: f"While unconstrained \\texttt{{Q-Merge (Adam STE)}} recovers some performance (\\textbf{{{qm_4_avg_mean:.2f}\\%}}), it remains limited by high-dimensional overfitting to calibration noise. Our proposed \\textbf{{Q-PolyMerge (Adam STE)}} achieves the highest performance at \\textbf{{{pm_4_avg_mean:.2f}\\%}}, representing a significant \\textbf{{+{pm_4_avg_mean - qm_4_avg_mean:.2f}\\%}} absolute improvement over the unconstrained baseline and a \\textbf{{+{pm_4_avg_mean - u_4_avg_mean:.2f}\\%}} absolute improvement over naive post-merge quantization.",
        tex_content,
        flags=re.DOTALL
    )
    tex_content = re.sub(
        r"full-precision \\texttt\{AdaMerging \(Adam \\to 4-Bit\)\} achieves \\textbf\{[0-9.]+\\\%\}\s+accuracy compared to \\texttt\{Q-PolyMerge \(Adam\)\} at \\textbf\{[0-9.]+\\\%\}\.",
        lambda m: f"full-precision \\texttt{{AdaMerging (Adam \\to 4-Bit)}} achieves \\textbf{{{am_4_adam_mean:.2f}\\%}} accuracy compared to \\texttt{{Q-PolyMerge (Adam)}} at \\textbf{{{pm_4_adam_mean:.2f}\\%}}.",
        tex_content
    )
    tex_content = re.sub(
        r"unconstrained search collapses to \\textbf\{[0-9.]+\\\%\}\s+for \\texttt\{AdaMerging \(ES \\to 4-Bit\)\}, while our proposed \\texttt\{Q-PolyMerge \(ES\)\} achieves a robust \\textbf\{[0-9.]+\\\%\}\s+average accuracy\s+\(a significant \\textbf\{([+-][0-9.]+)\\\%\}\s+absolute improvement\)",
        lambda m: f"unconstrained search collapses to \\textbf{{{am_4_es_mean:.2f}\\%}} for \\texttt{{AdaMerging (ES \\to 4-Bit)}}, while our proposed \\texttt{{Q-PolyMerge (ES)}} achieves a robust \\textbf{{{pm_4_es_mean:.2f}\\%}} average accuracy (a significant \\textbf{{+{pm_4_es_mean - am_4_es_mean:.2f}\\%}} absolute improvement)",
        tex_content
    )

    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"Successfully updated and synced all tables and references in {tex_path}.")

    # 4. Update appendix.tex
    app_path = "submission/sections/appendix.tex"
    with open(app_path, "r") as f:
        app_content = f.read()

    app_content = update_table_in_latex(app_content, "tab:unquantized_results", table1_mappings, results)
    app_content = update_table_in_latex(app_content, "tab:degree_ablation", table4_mappings, results)
    app_content = update_table_in_latex(app_content, "tab:blockwise_comparison", table5_mappings, results)

    # Parse key appendix narrative numbers
    d1_mean = results["Linear (d=1)"][4][0]
    d2_mean = results["Quadratic (d=2, Proposed)"][4][0]
    d3_mean = results["Cubic (d=3)"][4][0]
    d4_mean = results["Quartic (d=4)"][4][0]
    
    block_adam_mean = results["Block-wise Constant (Adam STE)"][4][0]
    pm_block_adam_mean = results["Polynomial Continuous (Adam STE, Ours)"][4][0]

    # Update appendix degree ablation text narrative
    app_content = re.sub(
        r"We observe that a linear trajectory \(\$d=1\$\) achieves an average test accuracy of \\textbf\{[0-9.]+\\\%\}\.\s+Increasing the degree to a quadratic polynomial \(\$d=2\$, our proposed default\) yields an average accuracy of \\textbf\{[0-9.]+\\\%\}\.\s+Further increasing the degree to cubic \(\$d=3\$\) or quartic \(\$d=4\$\) provides slightly higher accuracies \(\\textbf\{[0-9.]+\\\%\}\s+and \\textbf\{[0-9.]+\\\%\},\s+respectively\)\.",
        lambda m: f"We observe that a linear trajectory ($d=1$) achieves an average test accuracy of \\textbf{{{d1_mean:.2f}\\%}}. Increasing the degree to a quadratic polynomial ($d=2$, our proposed default) yields an average accuracy of \\textbf{{{d2_mean:.2f}\\%}}. Further increasing the degree to cubic ($d=3$) or quartic ($d=4$) provides slightly higher accuracies (\\textbf{{{d3_mean:.2f}\\%}} and \\textbf{{{d4_mean:.2f}\\%}}, respectively).",
        app_content
    )

    # Update appendix block-wise text narrative
    app_content = re.sub(
        r"Under Adam STE, Q-PolyMerge's continuous polynomial trajectory achieves \\textbf\{[0-9.]+\\\%\}\s+average accuracy, strictly outperforming Block-wise Constant scaling \(\\textbf\{[0-9.]+\\\%\}\)\s+by \\textbf\{([+-][0-9.]+)\\\%\}\s+absolute accuracy",
        lambda m: f"Under Adam STE, Q-PolyMerge's continuous polynomial trajectory achieves \\textbf{{{pm_block_adam_mean:.2f}\\%}} average accuracy, strictly outperforming Block-wise Constant scaling (\\textbf{{{block_adam_mean:.2f}\\%}}) by \\textbf{{+{pm_block_adam_mean - block_adam_mean:.2f}\\%}} absolute accuracy",
        app_content
    )

    with open(app_path, "w") as f:
        f.write(app_content)
    print(f"Successfully updated and synced all tables and references in {app_path}.")

if __name__ == "__main__":
    main()
