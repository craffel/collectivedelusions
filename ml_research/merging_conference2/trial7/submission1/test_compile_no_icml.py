import subprocess

latex_code = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}

\begin{document}

\begin{table*}[t]
\caption{Expert Parameter Drift}
\vskip 0.15in
\begin{center}
\small
\scshape
\begin{tabular}{lccccc}
\toprule
Metric & Scenario A & Scenario B & Scenario C & Scenario D & Scenario E \\
\midrule
MNIST Drift & 0.7371 $\pm$ 0.0072 & 0.7993 $\pm$ 0.0089 & 3.2996 $\pm$ 0.0992 & 22.1812 $\pm$ 0.3835 & 3.3193 $\pm$ 0.1056 \\
\bottomrule
\end{tabular}
\end{center}
\end{table*}

\end{document}
"""

with open("test_table_no_icml.tex", "w") as f:
    f.write(latex_code)

print("Compiling test_table_no_icml.tex...")
res = subprocess.run(["tectonic", "test_table_no_icml.tex"], capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("Exit Code:", res.returncode)
