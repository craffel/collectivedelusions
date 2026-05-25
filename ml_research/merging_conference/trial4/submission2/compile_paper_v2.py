import json
import os
import subprocess

def compile_paper():
    results_path = './experts/evaluation_results.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Evaluation results are not ready yet.")
        return False
        
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    tex_path = 'submission.tex'
    if not os.path.exists(tex_path):
        print(f"Error: {tex_path} not found!")
        return False
        
    with open(tex_path, 'r') as f:
        content = f.read()
        
    # Generate the LaTeX results table with mean +- std dev using $\pm$
    table_latex = """\\begin{table*}[t]
\\caption{Multi-task average accuracy (\\%) across 5 random seeds under different test-time environments and corruptions on the MNIST-FashionMNIST-KMNIST benchmark. Standard deviation is reported across the 5 seeds.}
\\label{tab:results}
\\begin{center}
\\begin{small}
\\begin{sc}
\\begin{tabular}{lccccc}
\\toprule
Method & Clean & Gaussian Noise & Gaussian Blur & Contrast & Brightness \\\\
\\midrule
Static Merged & """ + f"{results['clean']['static_merged']['mean']:.2f}\\% $\\pm$ {results['clean']['static_merged']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_noise']['static_merged']['mean']:.2f}\\% $\\pm$ {results['gaussian_noise']['static_merged']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_blur']['static_merged']['mean']:.2f}\\% $\\pm$ {results['gaussian_blur']['static_merged']['std']:.2f}\\%" + """ & """ + f"{results['contrast']['static_merged']['mean']:.2f}\\% $\\pm$ {results['contrast']['static_merged']['std']:.2f}\\%" + """ & """ + f"{results['brightness']['static_merged']['mean']:.2f}\\% $\\pm$ {results['brightness']['static_merged']['std']:.2f}\\%" + """ \\\\
\\midrule
\\textbf{Teacher-Guided (Heavy):} \\\\
Standard TTA (TG) & """ + f"{results['clean']['standard_tta_tg']['mean']:.2f}\\% $\\pm$ {results['clean']['standard_tta_tg']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_noise']['standard_tta_tg']['mean']:.2f}\\% $\\pm$ {results['gaussian_noise']['standard_tta_tg']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_blur']['standard_tta_tg']['mean']:.2f}\\% $\\pm$ {results['gaussian_blur']['standard_tta_tg']['std']:.2f}\\%" + """ & """ + f"{results['contrast']['standard_tta_tg']['mean']:.2f}\\% $\\pm$ {results['contrast']['standard_tta_tg']['std']:.2f}\\%" + """ & """ + f"{results['brightness']['standard_tta_tg']['mean']:.2f}\\% $\\pm$ {results['brightness']['standard_tta_tg']['std']:.2f}\\%" + """ \\\\
EWC-TTA (TG) & """ + f"{results['clean']['ewc_tta']['mean']:.2f}\\% $\\pm$ {results['clean']['ewc_tta']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_noise']['ewc_tta']['mean']:.2f}\\% $\\pm$ {results['gaussian_noise']['ewc_tta']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_blur']['ewc_tta']['mean']:.2f}\\% $\\pm$ {results['gaussian_blur']['ewc_tta']['std']:.2f}\\%" + """ & """ + f"{results['contrast']['ewc_tta']['mean']:.2f}\\% $\\pm$ {results['contrast']['ewc_tta']['std']:.2f}\\%" + """ & """ + f"{results['brightness']['ewc_tta']['mean']:.2f}\\% $\\pm$ {results['brightness']['ewc_tta']['std']:.2f}\\%" + """ \\\\
\\midrule
\\textbf{Teacher-Free (Light):} \\\\
Standard TTA (TF) & """ + f"{results['clean']['standard_tta_tf']['mean']:.2f}\\% $\\pm$ {results['clean']['standard_tta_tf']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_noise']['standard_tta_tf']['mean']:.2f}\\% $\\pm$ {results['gaussian_noise']['standard_tta_tf']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_blur']['standard_tta_tf']['mean']:.2f}\\% $\\pm$ {results['gaussian_blur']['standard_tta_tf']['std']:.2f}\\%" + """ & """ + f"{results['contrast']['standard_tta_tf']['mean']:.2f}\\% $\\pm$ {results['contrast']['standard_tta_tf']['std']:.2f}\\%" + """ & """ + f"{results['brightness']['standard_tta_tf']['mean']:.2f}\\% $\\pm$ {results['brightness']['standard_tta_tf']['std']:.2f}\\%" + """ \\\\
S2C-Merge (TF) & """ + f"{results['clean']['s2c_merge']['mean']:.2f}\\% $\\pm$ {results['clean']['s2c_merge']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_noise']['s2c_merge']['mean']:.2f}\\% $\\pm$ {results['gaussian_noise']['s2c_merge']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_blur']['s2c_merge']['mean']:.2f}\\% $\\pm$ {results['gaussian_blur']['s2c_merge']['std']:.2f}\\%" + """ & """ + f"{results['contrast']['s2c_merge']['mean']:.2f}\\% $\\pm$ {results['contrast']['s2c_merge']['std']:.2f}\\%" + """ & """ + f"{results['brightness']['s2c_merge']['mean']:.2f}\\% $\\pm$ {results['brightness']['s2c_merge']['std']:.2f}\\%" + """ \\\\
\\midrule
\\textbf{Ours (Teacher-Free):} \\\\
UEWC-Merge (no EWC) & """ + f"{results['clean']['uewc_merge_no_ewc']['mean']:.2f}\\% $\\pm$ {results['clean']['uewc_merge_no_ewc']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_noise']['uewc_merge_no_ewc']['mean']:.2f}\\% $\\pm$ {results['gaussian_noise']['uewc_merge_no_ewc']['std']:.2f}\\%" + """ & """ + f"{results['gaussian_blur']['uewc_merge_no_ewc']['mean']:.2f}\\% $\\pm$ {results['gaussian_blur']['uewc_merge_no_ewc']['std']:.2f}\\%" + """ & """ + f"{results['contrast']['uewc_merge_no_ewc']['mean']:.2f}\\% $\\pm$ {results['contrast']['uewc_merge_no_ewc']['std']:.2f}\\%" + """ & """ + f"{results['brightness']['uewc_merge_no_ewc']['mean']:.2f}\\% $\\pm$ {results['brightness']['uewc_merge_no_ewc']['std']:.2f}\\%" + """ \\\\
\\textbf{UEWC-Merge (Ours)} & \\textbf{""" + f"{results['clean']['uewc_merge']['mean']:.2f}\\%" + """} $\\pm$ \\textbf{""" + f"{results['clean']['uewc_merge']['std']:.2f}\\%" + """} & \\textbf{""" + f"{results['gaussian_noise']['uewc_merge']['mean']:.2f}\\%" + """} $\\pm$ \\textbf{""" + f"{results['gaussian_noise']['uewc_merge']['std']:.2f}\\%" + """} & \\textbf{""" + f"{results['gaussian_blur']['uewc_merge']['mean']:.2f}\\%" + """} $\\pm$ \\textbf{""" + f"{results['gaussian_blur']['uewc_merge']['std']:.2f}\\%" + """} & \\textbf{""" + f"{results['contrast']['uewc_merge']['mean']:.2f}\\%" + """} $\\pm$ \\textbf{""" + f"{results['contrast']['uewc_merge']['std']:.2f}\\%" + """} & \\textbf{""" + f"{results['brightness']['uewc_merge']['mean']:.2f}\\%" + """} $\\pm$ \\textbf{""" + f"{results['brightness']['uewc_merge']['std']:.2f}\\%" + """} \\\\
\\bottomrule
\\end{tabular}
\\end{sc}
\\end{small}
\\end{center}
\\vskip -0.1in
\\end{table*}"""

    # Inject the results table and the plot
    content_updated = content.replace('% RESULTS_TABLE_PLACEHOLDER', table_latex)
    
    # We should also insert the plot in the Results and Analysis section
    # Let's find where \subsection{Empirical Results} is and insert the figure block
    fig_latex = """\\begin{figure*}[t]
  \\vskip 0.2in
  \\begin{center}
    \\centerline{\\includegraphics[width=1.9\\columnwidth]{experts/results_plot}}
    \\caption{Comparison of Multi-Task Average Accuracy (\\%) across different Test-Time environments. UEWC-Merge (Ours) stabilizes classification heads under self-supervised objectives, decisively outperforming unconstrained Standard TTA and frozen-classifier S2C-Merge.}
    \\label{fig:results_plot}
  \\end{center}
\\end{figure*}"""

    content_updated = content_updated.replace('\\subsection{Empirical Results}', '\\subsection{Empirical Results}\n' + fig_latex)
    
    with open(tex_path, 'w') as f:
        f.write(content_updated)
        
    print("Successfully populated results table and figure in submission.tex!")
    
    # Run Tectonic to compile
    print("Compiling paper with Tectonic...")
    tectonic_path = '/fsx/craffel/miniconda3/bin/tectonic'
    result = subprocess.run([tectonic_path, 'submission.tex'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Tectonic compilation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
        
    print("Tectonic compiled successfully!")
    
    # Copy output to submission.pdf if it compiled under another name
    if os.path.exists('submission.pdf'):
        print("Generated submission.pdf is available in the root folder!")
        return True
    else:
        print("Error: submission.pdf was not found after compilation.")
        return False

if __name__ == '__main__':
    compile_paper()
