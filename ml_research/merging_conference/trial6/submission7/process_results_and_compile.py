import os
import json
import time
import shutil
import subprocess

def wait_for_results(json_path, timeout=600, check_interval=15):
    print(f"Waiting for results file at {json_path} (timeout: {timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(json_path):
            # Check if file has been completely written (not empty)
            if os.path.getsize(json_path) > 0:
                try:
                    with open(json_path, "r") as f:
                        json.load(f)
                    print(f"Results file found and successfully loaded after {time.time() - start_time:.1f}s.")
                    return True
                except json.JSONDecodeError:
                    pass
        time.sleep(check_interval)
    print("Timeout reached while waiting for results.")
    return False

def main():
    json_path = "results/ttmm_results.json"
    paper_path = "template/example_paper.tex"
    
    # 1. Wait for results to be written
    if not wait_for_results(json_path):
        print("Error: Could not find experimental results. Exiting.")
        return
        
    # 2. Read results
    with open(json_path, "r") as f:
        results = json.load(f)
        
    print("\n--- Experimental Results Loaded ---")
    
    # Extract average accuracies
    try:
        u_seq = results["Sequential"]["Uniform"]["avg_acc"]
        u_alt = results["Alternating"]["Uniform"]["avg_acc"]
        
        ada_seq = results["Sequential"]["AdaMerging"]["avg_acc"]
        ada_alt = results["Alternating"]["AdaMerging"]["avg_acc"]
        
        fpca_seq = results["Sequential"]["FP-CA"]["avg_acc"]
        fpca_alt = results["Alternating"]["FP-CA"]["avg_acc"]
        
        iggs_seq = results["Sequential"]["IGGS-Merge"]["avg_acc"]
        iggs_alt = results["Alternating"]["IGGS-Merge"]["avg_acc"]
        
        fpca_tms_seq = results["Sequential"]["FP-CA + TMS (Ours)"]["avg_acc"]
        fpca_tms_alt = results["Alternating"]["FP-CA + TMS (Ours)"]["avg_acc"]
        
        iggs_tms_seq = results["Sequential"]["IGGS-Merge + TMS (Ours)"]["avg_acc"]
        iggs_tms_alt = results["Alternating"]["IGGS-Merge + TMS (Ours)"]["avg_acc"]
    except KeyError as e:
        print(f"Error: Missing expected result key in JSON: {e}")
        return
        
    # Print comparison table in text
    print("\n" + "="*80)
    print(f"{'Algorithm':<25} | {'Sequential Acc':<18} | {'Alternating Acc':<18}")
    print("="*80)
    print(f"{'Uniform':<25} | {u_seq:<18.2f} | {u_alt:<18.2f}")
    print(f"{'AdaMerging':<25} | {ada_seq:<18.2f} | {ada_alt:<18.2f}")
    print(f"{'FP-CA':<25} | {fpca_seq:<18.2f} | {fpca_alt:<18.2f}")
    print(f"{'IGGS-Merge':<25} | {iggs_seq:<18.2f} | {iggs_alt:<18.2f}")
    print(f"{'FP-CA + TMS (Ours)':<25} | {fpca_tms_seq:<18.2f} | {fpca_tms_alt:<18.2f}")
    print(f"{'IGGS-Merge + TMS (Ours)':<25} | {iggs_tms_seq:<18.2f} | {iggs_tms_alt:<18.2f}")
    print("="*80)
    
    # 3. Create the LaTeX table string using raw f-string and properly doubled curly braces
    latex_table = fr"""\begin{{table}}[t]
\caption{{Test-time model merging adaptation average accuracy (\%) on Sequential and Alternating streams using MNIST, FashionMNIST, and KMNIST expert models. Bold indicates best performance.}}
\label{{tab:main_results}}
\vskip 0.15in
\begin{{center}}
\begin{{small}}
\begin{{sc}}
\begin{{tabular}}{{lcc}}
\toprule
Algorithm & Sequential Acc (\%) & Alternating Acc (\%) \\
\midrule
Uniform & {u_seq:.2f} & {u_alt:.2f} \\
AdaMerging & {ada_seq:.2f} & {ada_alt:.2f} \\
FP-CA & {fpca_seq:.2f} & {fpca_alt:.2f} \\
IGGS-Merge & {iggs_seq:.2f} & {iggs_alt:.2f} \\
\midrule
FP-CA + TMS (Ours) & \textbf{{{fpca_tms_seq:.2f}}} & {fpca_tms_alt:.2f} \\
IGGS-Merge + TMS (Ours) & {iggs_tms_seq:.2f} & \textbf{{{iggs_tms_alt:.2f}}} \\
\bottomrule
\end{{tabular}}
\end{{sc}}
\end{{small}}
\end{{center}}
\vskip -0.1in
\end{{table}}"""

    # 4. Modify template/example_paper.tex
    if not os.path.exists(paper_path):
        print(f"Error: {paper_path} not found.")
        return
        
    print(f"Reading {paper_path}...")
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    if "% MAIN_RESULTS_TABLE_PLACEHOLDER" in content:
        print("Inserting results table into LaTeX paper...")
        content = content.replace("% MAIN_RESULTS_TABLE_PLACEHOLDER", latex_table)
        with open(paper_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Successfully updated paper with results table.")
    else:
        print("Warning: Placeholder '% MAIN_RESULTS_TABLE_PLACEHOLDER' not found in LaTeX file.")
        
    # 5. Compile the LaTeX file to PDF using tectonic
    tectonic_bin = "/fsx/craffel/miniconda3/bin/tectonic"
    print(f"Compiling {paper_path} using Tectonic...")
    try:
        result = subprocess.run([tectonic_bin, paper_path], capture_output=True, text=True, check=True)
        print("Tectonic compilation completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Tectonic compilation failed with error:")
        print(e.stderr)
        print(e.stdout)
        return
        
    # 6. Copy the output PDF to submission.pdf in current directory
    compiled_pdf = "template/example_paper.pdf"
    target_pdf = "submission.pdf"
    if os.path.exists(compiled_pdf):
        shutil.copy(compiled_pdf, target_pdf)
        print(f"Copied compiled PDF to {target_pdf} successfully!")
    else:
        print(f"Error: Compiled PDF {compiled_pdf} not found.")

if __name__ == "__main__":
    main()
