import json

def format_entry(mean, std, is_best):
    val_str = f"{mean:.2f} \pm {std:.2f}"
    if is_best:
        return f"$\\mathbf{{{val_str}}}$"
    else:
        return f"${val_str}$"

def generate_table(loss_mode, title, label, filename, results):
    environments = ["Clean", "Gaussian Noise", "Gaussian Blur", "Contrast"]
    streams = ["Alternating", "Sequential"]
    methods = ["Static", "Static Fisher", "Standard TTA", "FiT-Merge (Ours)"]
    
    # Header of the table
    out = []
    out.append("\\begin{table*}[t]")
    out.append(f"\\caption{{{title}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\vskip 0.05in")
    out.append("\\begin{center}")
    out.append("\\begin{scriptsize}")
    out.append("\\begin{sc}")
    out.append("\\setlength{\\tabcolsep}{2.5pt}")
    out.append("\\begin{tabular}{lcccccccc}")
    out.append("\\toprule")
    out.append("& \\multicolumn{4}{c}{\\textbf{Alternating Stream}} & \\multicolumn{4}{c}{\\textbf{Sequential Stream}} \\\\")
    out.append("\\cmidrule(r){2-5} \\cmidrule(l){6-9}")
    out.append("Method & Clean & Noise & Blur & Contrast & Clean & Noise & Blur & Contrast \\\\")
    out.append("\\midrule")
    
    # For each method, build the row
    for method in methods:
        row_entries = [method]
        for stream in streams:
            # Find the best mean in this stream + env to bold it
            for env in environments:
                # Find which method is best for this specific (stream, env)
                best_mean = -1.0
                best_method = ""
                for m in methods:
                    m_mean = results[loss_mode][stream][env][m]["mean"]
                    if m_mean > best_mean:
                        best_mean = m_mean
                        best_method = m
                
                mean = results[loss_mode][stream][env][method]["mean"]
                std = results[loss_mode][stream][env][method]["std"]
                is_best = (method == best_method)
                row_entries.append(format_entry(mean, std, is_best))
        
        out.append(" & ".join(row_entries) + " \\\\")
        
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{sc}")
    out.append("\\end{scriptsize}")
    out.append("\\end{center}")
    out.append("\\vskip -0.15in")
    out.append("\\end{table*}")
    
    with open(filename, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"Successfully generated {filename}")

def main():
    with open("multi_seed_results.json", "r") as f:
        results = json.load(f)
        
    generate_table(
        loss_mode="teacher-supervised",
        title="Multi-task average accuracy (\\%) across different environmental corruptions under the \\textbf{teacher-supervised (KL loss)} test-time adaptation setting over 5 independent random seeds.",
        label="tab:results_supervised",
        filename="table_results_supervised.tex",
        results=results
    )
    
    generate_table(
        loss_mode="teacher-free",
        title="Multi-task average accuracy (\\%) across different environmental corruptions under the \\textbf{teacher-free (entropy minimization)} test-time adaptation setting over 5 independent random seeds.",
        label="tab:results_free",
        filename="table_results_free.tex",
        results=results
    )

if __name__ == "__main__":
    main()
