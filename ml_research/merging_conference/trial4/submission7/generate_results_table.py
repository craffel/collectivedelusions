import json
import os

def generate_tables():
    if not os.path.exists("tta_results.json"):
        print("tta_results.json not found. Cannot generate tables.")
        return
        
    with open("tta_results.json", "r") as f:
        results = json.load(f)
        
    environments = ["Clean", "Gaussian Noise", "Gaussian Blur", "Contrast"]
    methods = ["Static", "Static Fisher", "Standard TTA", "FiT-Merge (Ours)"]
    streams = ["Alternating", "Sequential"]
    
    # 1. Supervised Table
    supervised_tex = r"""\begin{table*}[t]
\caption{Multi-task average accuracy (\%) across different environmental corruptions under the \textbf{teacher-supervised (KL loss)} test-time adaptation setting.}
\label{tab:results_supervised}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{4}{c}{\textbf{Alternating Stream}} & \multicolumn{4}{c}{\textbf{Sequential Stream}} \\
\cmidrule(r){2-5} \cmidrule(l){6-9}
Method & Clean & Noise & Blur & Contrast & Clean & Noise & Blur & Contrast \\
\midrule
"""
    for method in methods:
        row_cells = [method]
        for stream_type in streams:
            for env in environments:
                acc = results["teacher-supervised"][stream_type][env][method]["accuracy"]
                row_cells.append(f"{acc:.2f}")
        supervised_tex += " & ".join(row_cells) + " \\\\\n"
        
    supervised_tex += r"""\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table*}
"""
    with open("table_results_supervised.tex", "w") as f:
        f.write(supervised_tex)
    print("Generated table_results_supervised.tex")
    
    # 2. Teacher-Free Table
    free_tex = r"""\begin{table*}[t]
\caption{Multi-task average accuracy (\%) across different environmental corruptions under the \textbf{teacher-free (entropy minimization)} test-time adaptation setting.}
\label{tab:results_free}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{4}{c}{\textbf{Alternating Stream}} & \multicolumn{4}{c}{\textbf{Sequential Stream}} \\
\cmidrule(r){2-5} \cmidrule(l){6-9}
Method & Clean & Noise & Blur & Contrast & Clean & Noise & Blur & Contrast \\
\midrule
"""
    for method in methods:
        row_cells = [method]
        for stream_type in streams:
            for env in environments:
                acc = results["teacher-free"][stream_type][env][method]["accuracy"]
                row_cells.append(f"{acc:.2f}")
        free_tex += " & ".join(row_cells) + " \\\\\n"
        
    free_tex += r"""\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table*}
"""
    with open("table_results_free.tex", "w") as f:
        f.write(free_tex)
    print("Generated table_results_free.tex")

if __name__ == "__main__":
    generate_tables()
