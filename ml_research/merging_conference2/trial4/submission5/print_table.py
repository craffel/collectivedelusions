import json

def print_latex_table():
    try:
        with open("results.json") as f:
            res = json.load(f)
    except FileNotFoundError:
        print("results.json not found yet.")
        return

    methods = ['none', 'sp-taac', 'r-taac', 's-tcac', 'taac', 'fwas', 'reda']
    cal_sizes = ['4', '8', '16', '32', '64', '128', '256']
    
    # Method names map
    method_names = {
        'none': 'None (No Calib)',
        'sp-taac': 'SP-TAAC',
        'r-taac': 'R-TAAC',
        's-tcac': 'S-TCAC',
        'taac': 'TAAC',
        'fwas': 'FWAS (Ours)',
        'reda': 'REDA'
    }
    
    print("\\begin{tabular}{lccccccc}")
    print("\\toprule")
    print("Method & $N=4$ & $N=8$ & $N=16$ & $N=32$ & $N=64$ & $N=128$ & $N=256$ \\\\")
    print("\\midrule")
    
    # We want to highlight the best method for each N in bold
    # To do that, we find the best method for each N (excluding REDA at low N or including it)
    best_per_n = {}
    for size in cal_sizes:
        best_val = -1
        best_method = None
        for m in methods:
            if size in res[m] and 'avg' in res[m][size]:
                val = res[m][size]['avg']
                if val > best_val:
                    best_val = val
                    best_method = m
        best_per_n[size] = best_method
        
    for m in methods:
        row_str = f"{method_names[m]:<15}"
        for size in cal_sizes:
            if size in res[m] and 'avg' in res[m][size]:
                val = res[m][size]['avg']
                if best_per_n[size] == m:
                    row_str += f" & \\textbf{{{val:.2f}}}"
                else:
                    row_str += f" & {val:.2f}"
            else:
                row_str += " & -"
        row_str += " \\\\"
        print(row_str)
        
    print("\\bottomrule")
    print("\\end{tabular}")

if __name__ == "__main__":
    print_latex_table()
