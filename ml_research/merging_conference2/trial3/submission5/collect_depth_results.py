import os
import glob

def parse_csv_from_file(filepath):
    lines = []
    started = False
    with open(filepath, "r") as f:
        for line in f:
            if "RESULTS_CSV_START" in line:
                started = True
                continue
            if "RESULTS_CSV_END" in line:
                started = False
                break
            if started:
                lines.append(line.strip())
    return lines

def main():
    out_files = sorted(glob.glob("sweep_depth_*.out"))
    print(f"Found {len(out_files)} depth sweep log files.")
    
    results = {}
    
    for filepath in out_files:
        name = os.path.basename(filepath).replace("sweep_depth_", "").replace(".out", "")
        csv_lines = parse_csv_from_file(filepath)
        if not csv_lines:
            continue
            
        # Parse CSV lines
        # Method,MNIST,Fashion,CIFAR10,Average
        run_data = {}
        for line in csv_lines[1:]: # skip header
            parts = line.split(",")
            method = parts[0]
            mnist = float(parts[1])
            fashion = float(parts[2])
            cifar = float(parts[3])
            avg = float(parts[4])
            run_data[method] = (mnist, fashion, cifar, avg)
            
        results[name] = run_data
        
    if not results:
        print("No results parsed yet. Are the sweeps still running?")
        return
        
    print("\n" + "="*80)
    print("ADAPTATION DEPTH ABLATION RESULTS (N=128, WA, Seed=42)")
    print("="*80)
    print(f"{'Adapt Depth':<12} | {'Method':<12} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR10':<8} | {'Average':<8}")
    print("-" * 80)
    
    for depth in ["heads", "layer4", "full"]:
        if depth not in results:
            continue
        data = results[depth]
        for method in ["Head-SFT", "Head-TTA", "SPJA-SFT", "SPJA-TTA"]:
            if method in data:
                m, f, c, a = data[method]
                print(f"{depth:<12} | {method:<12} | {m:<8.2f} | {f:<8.2f} | {c:<8.2f} | {a:<8.2f}")
        print("-" * 80)

    # Let's also print a beautiful LaTeX table block
    print("\n" + "="*80)
    print("LATEX TABLE CODE FOR SUBMISSION.TEX")
    print("="*80)
    print(r"""\begin{table}[h]
\caption{Adaptation depth ablation study under Weight Averaging (WA) with calibration size $N=128$. We compare adapting only classification Heads, Layer 4 + Heads, and the Full backbone + Heads.}
\label{tab:adaptation_depth}
\begin{center}
\begin{small}
\begin{sc}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Method / Depth & MNIST & Fashion-MNIST & CIFAR-10 & Average \\
\midrule""")
    
    for depth in ["heads", "layer4", "full"]:
        print(f"\\multicolumn{{5}}{{l}}{{\\textbf{{Adaptation Depth: {depth.capitalize()}}}}} \\\\")
        print(r"\midrule")
        data = results.get(depth, {})
        for method in ["Head-SFT", "Head-TTA", "SPJA-SFT", "SPJA-TTA"]:
            if method in data:
                m, f, c, a = data[method]
                method_name = method
                if method == "SPJA-SFT":
                    method_name = "SPJA-SFT (Ours)"
                elif method == "SPJA-TTA":
                    method_name = "\\textbf{SPJA-TTA (Ours)}"
                print(f"{method_name} & {m:.2f} & {f:.2f} & {c:.2f} & {a:.2f} \\\\")
        print(r"\midrule")
        
    print(r"""\bottomrule
\end{tabular}%
}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}""")

if __name__ == "__main__":
    main()
