import re
import sys

def clean_val(val):
    val = val.strip().replace('%', '')
    if '±' in val:
        mean, std = val.split('±')
        mean = mean.strip()
        std = std.strip()
        # If std is extremely small, e.g., 0.00, we can choose to omit it or print it
        if float(std) == 0.0:
            return f"{mean}"
        return f"{mean} \\pm {std}"
    return val

def parse_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Find the main table
    main_pattern = r"FINAL MULTI-SEED AGGREGATED RESULTS TABLE\n=+\n.*?\n-+\n(.*?)\n=+"
    main_match = re.search(main_pattern, content, re.DOTALL)
    
    if not main_match:
        print("Main aggregated table not found in output file yet.")
        return None, None, None

    main_rows_raw = main_match.group(1).strip().split('\n')
    main_rows = []
    for row in main_rows_raw:
        parts = [p.strip() for p in row.split('|')]
        if len(parts) == 7:
            n_val, method, mnist, fashion, cifar, mean, sft = parts
            n_val = n_val.strip()
            method_name = method.strip()
            if method_name == 'HSC':
                method_name = "\\textbf{HSC (Ours)}"
            elif method_name == 'uncalibrated':
                method_name = "Uncalibrated (WA)"
            
            # Format row
            mnist_fmt = clean_val(mnist)
            fashion_fmt = clean_val(fashion)
            cifar_fmt = clean_val(cifar)
            mean_fmt = clean_val(mean)
            sft_fmt = clean_val(sft)
            
            # Add bold to HSC
            if 'HSC' in method_name:
                mean_fmt = f"\\textbf{{{mean_fmt}}}"
                sft_fmt = f"\\textbf{{{sft_fmt}}}"
                
            main_rows.append((n_val, method_name, mnist_fmt, fashion_fmt, cifar_fmt, mean_fmt, sft_fmt))

    # Find split index sweep table
    split_pattern = r"HSC SPLIT SWEEP \(N=128\)\n=+\n.*?\n-+\n(.*?)\n=+"
    split_match = re.search(split_pattern, content, re.DOTALL)
    split_rows = []
    if split_match:
        split_rows_raw = split_match.group(1).strip().split('\n')
        for row in split_rows_raw:
            parts = [p.strip() for p in row.split('|')]
            if len(parts) == 3:
                split_idx, mean, sft = parts
                split_idx_val = split_idx.strip()
                mean_fmt = clean_val(mean)
                sft_fmt = clean_val(sft)
                
                label = split_idx_val
                if split_idx_val == '0':
                    label = "0 (Pure R-TAAC)"
                elif split_idx_val == '15':
                    label = "\\textbf{15 (Layer 4 Block)}"
                    mean_fmt = f"\\textbf{{{mean_fmt}}}"
                    sft_fmt = f"\\textbf{{{sft_fmt}}}"
                    
                split_rows.append((label, mean_fmt, sft_fmt))

    # Find alpha sweep table
    alpha_pattern = r"HSC ALPHA SWEEP \(N=128, Split=15\)\n=+\n.*?\n-+\n(.*?)\n=+"
    alpha_match = re.search(alpha_pattern, content, re.DOTALL)
    alpha_rows = []
    if alpha_match:
        alpha_rows_raw = alpha_match.group(1).strip().split('\n')
        for row in alpha_rows_raw:
            parts = [p.strip() for p in row.split('|')]
            if len(parts) == 3:
                alpha_val, mean, sft = parts
                alpha_val_str = alpha_val.strip()
                mean_fmt = clean_val(mean)
                sft_fmt = clean_val(sft)
                
                label = alpha_val_str
                if float(alpha_val_str) == 0.25:
                    label = "\\textbf{0.25}"
                    mean_fmt = f"\\textbf{{{mean_fmt}}}"
                    sft_fmt = f"\\textbf{{{sft_fmt}}}"
                    
                alpha_rows.append((label, mean_fmt, sft_fmt))

    return main_rows, split_rows, alpha_rows

def generate_latex_main_table(main_rows):
    if not main_rows:
        return ""
    
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\caption{Multi-task model merging accuracies (\\%) on ResNet-18 across calibration sizes $N$ (aggregated over multiple seeds). Methods are evaluated both without head SFT (Representation only) and with head SFT (Mean+SFT).}")
    latex.append("\\label{tab:main}")
    latex.append("\\vskip 0.15in")
    latex.append("\\begin{center}")
    latex.append("\\begin{small}")
    latex.append("\\begin{tabular}{llccccc}")
    latex.append("\\toprule")
    latex.append("N & Method & MNIST Acc & F-MNIST Acc & CIFAR-10 Acc & Mean Acc & Mean+SFT Acc \\\\")
    latex.append("\\midrule")
    latex.append("- & Expert Models (Upper Bound) & 98.45 & 85.40 & 67.38 & 83.74 & -- \\\\")
    latex.append("\\midrule")
    
    prev_n = None
    for row in main_rows:
        n_val, method, mnist, fashion, cifar, mean, sft = row
        if prev_n is not None and prev_n != n_val:
            latex.append("\\midrule")
        
        latex.append(f"{n_val} & {method} & {mnist} & {fashion} & {cifar} & {mean} & {sft} \\\\")
        prev_n = n_val
        
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{small}")
    latex.append("\\end{center}")
    latex.append("\\vskip -0.15in")
    latex.append("\\vspace{-3mm}")
    latex.append("\\end{table*}")
    
    return "\n".join(latex)

def generate_latex_split_table(split_rows):
    if not split_rows:
        return ""
    latex = []
    latex.append("\\begin{tabular}{ccc}")
    latex.append("\\toprule")
    latex.append("Split Index ($M$) & Mean Acc (\\%) & Mean+SFT Acc (\\%) \\\\")
    latex.append("\\midrule")
    for row in split_rows:
        label, mean, sft = row
        latex.append(f"{label} & {mean} & {sft} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    return "\n".join(latex)

def generate_latex_alpha_table(alpha_rows):
    if not alpha_rows:
        return ""
    latex = []
    latex.append("\\begin{tabular}{ccc}")
    latex.append("\\toprule")
    latex.append("Alpha ($\\alpha$) & Mean Acc (\\%) & Mean+SFT Acc (\\%) \\\\")
    latex.append("\\midrule")
    for row in alpha_rows:
        label, mean, sft = row
        latex.append(f"{label} & {mean} & {sft} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    return "\n".join(latex)

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_and_generate_latex.py <logfile>")
        sys.exit(1)
        
    logfile = sys.argv[1]
    main_rows, split_rows, alpha_rows = parse_file(logfile)
    
    if main_rows is None:
        sys.exit(1)
        
    main_latex = generate_latex_main_table(main_rows)
    split_latex = generate_latex_split_table(split_rows)
    alpha_latex = generate_latex_alpha_table(alpha_rows)
    
    print("\n" + "="*40 + "\nLATEX MAIN TABLE\n" + "="*40)
    print(main_latex)
    print("\n" + "="*40 + "\nLATEX SPLIT TABLE\n" + "="*40)
    print(split_latex)
    print("\n" + "="*40 + "\nLATEX ALPHA TABLE\n" + "="*40)
    print(alpha_latex)

if __name__ == '__main__':
    main()
