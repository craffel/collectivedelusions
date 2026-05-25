import json

def regenerate_multi_seed_tables():
    with open("multi_seed_results.json", "r") as f:
        results = json.load(f)
        
    def format_entry(mean, std, is_best):
        val_str = f"{mean:.2f} \\pm {std:.2f}"
        if is_best:
            return f"$\\mathbf{{{val_str}}}$"
        else:
            return f"${val_str}$"

    def generate_table(loss_mode, title, label, filename, results):
        environments = ["Clean", "Gaussian Noise", "Gaussian Blur", "Contrast"]
        streams = ["Alternating", "Sequential"]
        methods = ["Static", "Static Fisher", "Standard TTA", "FiT-Merge (Ours)"]
        
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
        
        for method in methods:
            row_entries = [method]
            for stream in streams:
                for env in environments:
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

def regenerate_ablation_alpha():
    with open("ablation_alpha_results.json", "r") as f:
        ablation_results = json.load(f)
        
    alpha_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    latex = """\\begin{table}[h]
\\caption{Ablation study on the effect of the Fisher modulation exponent $\\alpha$ under the Alternating Stream setting. We report multi-task average accuracy (\\%). Note that $\\alpha = 0.0$ corresponds to standard layer-wise TTA.}
\\label{tab:ablation_alpha}
\\vskip 0.05in
\\begin{center}
\\begin{scriptsize}
\\begin{sc}
\\begin{tabular}{lcccc}
\\toprule
& \\multicolumn{2}{c}{\\textbf{Teacher-Supervised}} & \\multicolumn{2}{c}{\\textbf{Teacher-Free}} \\\\
\\cmidrule(r){2-3} \\cmidrule(l){4-5}
$\\alpha$ & Noise & Contrast & Noise & Contrast \\\\
\\midrule
"""
    for alpha in alpha_values:
        val_sup_noise = ablation_results["teacher-supervised"]["Alternating"]["Gaussian Noise"][str(alpha)]
        val_sup_contrast = ablation_results["teacher-supervised"]["Alternating"]["Contrast"][str(alpha)]
        val_free_noise = ablation_results["teacher-free"]["Alternating"]["Gaussian Noise"][str(alpha)]
        val_free_contrast = ablation_results["teacher-free"]["Alternating"]["Contrast"][str(alpha)]
        
        latex += f"{alpha:<5} & {val_sup_noise:.2f} & {val_sup_contrast:.2f} & {val_free_noise:.2f} & {val_free_contrast:.2f} \\\\\n"
        
    latex += """\\bottomrule
\\end{tabular}
\\end{sc}
\\end{scriptsize}
\\end{center}
\\vskip -0.15in
\\end{table}
"""
    with open("table_ablation_alpha.tex", "w") as f:
        f.write(latex)
    print("Successfully generated table_ablation_alpha.tex")

def regenerate_ablation_samples():
    with open("ablation_samples_results.json", "r") as f:
        ablation_results = json.load(f)
        
    sample_sizes = [10, 50, 100, 250, 500]
    
    latex = """\\begin{table*}[t]
\\caption{Ablation study on the effect of the number of samples used to estimate the diagonal Fisher Information Matrix (FIM) for FiT-Merge ($\\alpha = 0.1$). We report multi-task average accuracy (\\%) across Alternating and Sequential streams under Gaussian Noise and Contrast corruptions.}
\\label{tab:ablation_samples}
\\vskip 0.05in
\\begin{center}
\\begin{scriptsize}
\\begin{sc}
\\begin{tabular}{lcccccccc}
\\toprule
& \\multicolumn{4}{c}{\\textbf{Alternating Stream}} & \\multicolumn{4}{c}{\\textbf{Sequential Stream}} \\\\
\\cmidrule(r){2-5} \\cmidrule(l){6-9}
& \\multicolumn{2}{c}{\\textbf{Supervised}} & \\multicolumn{2}{c}{\\textbf{Teacher-Free}} & \\multicolumn{2}{c}{\\textbf{Supervised}} & \\multicolumn{2}{c}{\\textbf{Teacher-Free}} \\\\
\\cmidrule(r){2-3} \\cmidrule(l){4-5} \\cmidrule(r){6-7} \\cmidrule(l){8-9}
Samples & Noise & Contrast & Noise & Contrast & Noise & Contrast & Noise & Contrast \\\\
\\midrule
"""
    for size in sample_sizes:
        val_alt_sup_noise = ablation_results["teacher-supervised"]["Alternating"]["Gaussian Noise"][str(size)]
        val_alt_sup_contrast = ablation_results["teacher-supervised"]["Alternating"]["Contrast"][str(size)]
        val_alt_free_noise = ablation_results["teacher-free"]["Alternating"]["Gaussian Noise"][str(size)]
        val_alt_free_contrast = ablation_results["teacher-free"]["Alternating"]["Contrast"][str(size)]
        
        val_seq_sup_noise = ablation_results["teacher-supervised"]["Sequential"]["Gaussian Noise"][str(size)]
        val_seq_sup_contrast = ablation_results["teacher-supervised"]["Sequential"]["Contrast"][str(size)]
        val_seq_free_noise = ablation_results["teacher-free"]["Sequential"]["Gaussian Noise"][str(size)]
        val_seq_free_contrast = ablation_results["teacher-free"]["Sequential"]["Contrast"][str(size)]
        
        latex += f"{size:<7} & {val_alt_sup_noise:.2f} & {val_alt_sup_contrast:.2f} & {val_alt_free_noise:.2f} & {val_alt_free_contrast:.2f} & {val_seq_sup_noise:.2f} & {val_seq_sup_contrast:.2f} & {val_seq_free_noise:.2f} & {val_seq_free_contrast:.2f} \\\\\n"
        
    latex += """\\bottomrule
\\end{tabular}
\\end{sc}
\\end{scriptsize}
\\end{center}
\\vskip -0.15in
\\end{table*}
"""
    with open("table_ablation_samples.tex", "w") as f:
        f.write(latex)
    print("Successfully generated table_ablation_samples.tex")

def regenerate_ablation_epsilon():
    with open("ablation_epsilon_results.json", "r") as f:
        ablation_results = json.load(f)
        
    epsilon_values = [1e-08, 1e-06, 0.0001, 0.01, 1.0]
    
    latex = """\\begin{table*}[t]
\\caption{Ablation study on the effect of the smoothing epsilon $\\epsilon$ for FiT-Merge (with $\\alpha = 0.1$). We report multi-task average accuracy (\\%) across Alternating and Sequential streams under Gaussian Noise and Contrast corruptions.}
\\label{tab:ablation_epsilon}
\\vskip 0.05in
\\begin{center}
\\begin{scriptsize}
\\begin{sc}
\\begin{tabular}{lcccccccc}
\\toprule
& \\multicolumn{4}{c}{\\textbf{Alternating Stream}} & \\multicolumn{4}{c}{\\textbf{Sequential Stream}} \\\\
\\cmidrule(r){2-5} \\cmidrule(l){6-9}
& \\multicolumn{2}{c}{\\textbf{Supervised}} & \\multicolumn{2}{c}{\\textbf{Teacher-Free}} & \\multicolumn{2}{c}{\\textbf{Supervised}} & \\multicolumn{2}{c}{\\textbf{Teacher-Free}} \\\\
\\cmidrule(r){2-3} \\cmidrule(l){4-5} \\cmidrule(r){6-7} \\cmidrule(l){8-9}
$\\epsilon$ & Noise & Contrast & Noise & Contrast & Noise & Contrast & Noise & Contrast \\\\
\\midrule
"""
    for eps in epsilon_values:
        val_alt_sup_noise = ablation_results["teacher-supervised"]["Alternating"]["Gaussian Noise"][str(eps)]
        val_alt_sup_contrast = ablation_results["teacher-supervised"]["Alternating"]["Contrast"][str(eps)]
        val_alt_free_noise = ablation_results["teacher-free"]["Alternating"]["Gaussian Noise"][str(eps)]
        val_alt_free_contrast = ablation_results["teacher-free"]["Alternating"]["Contrast"][str(eps)]
        
        val_seq_sup_noise = ablation_results["teacher-supervised"]["Sequential"]["Gaussian Noise"][str(eps)]
        val_seq_sup_contrast = ablation_results["teacher-supervised"]["Sequential"]["Contrast"][str(eps)]
        val_seq_free_noise = ablation_results["teacher-free"]["Sequential"]["Gaussian Noise"][str(eps)]
        val_seq_free_contrast = ablation_results["teacher-free"]["Sequential"]["Contrast"][str(eps)]
        
        latex += f"{str(eps):<8} & {val_alt_sup_noise:.2f} & {val_alt_sup_contrast:.2f} & {val_alt_free_noise:.2f} & {val_alt_free_contrast:.2f} & {val_seq_sup_noise:.2f} & {val_seq_sup_contrast:.2f} & {val_seq_free_noise:.2f} & {val_seq_free_contrast:.2f} \\\\\n"
        
    latex += """\\bottomrule
\\end{tabular}
\\end{sc}
\\end{scriptsize}
\\end{center}
\\vskip -0.15in
\\end{table*}
"""
    with open("table_ablation_epsilon.tex", "w") as f:
        f.write(latex)
    print("Successfully generated table_ablation_epsilon.tex")

if __name__ == "__main__":
    regenerate_multi_seed_tables()
    regenerate_ablation_alpha()
    regenerate_ablation_samples()
    regenerate_ablation_epsilon()
