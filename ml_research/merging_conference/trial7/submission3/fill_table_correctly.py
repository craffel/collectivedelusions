import os

def main():
    results_dir = "results"
    methods = ["Static", "PROTO-TTMM", "IGGS-OW", "Rob-OW"]
    corruptions = ["clean", "corrupted"]
    
    # Parse actual results
    data = {}
    for method in methods:
        data[method] = {}
        for corr in corruptions:
            filepath = os.path.join(results_dir, f"result_{method}_{corr}_seed42.txt")
            data[method][corr] = {}
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} does not exist.")
                continue
            with open(filepath, "r") as f:
                content = f.read()
            for line in content.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if k in ["MNIST_Acc", "KMNIST_Acc", "FashionMNIST_Acc", "Overall_Acc", "NDR", "FPR"]:
                        data[method][corr][k] = float(v)

    # Read submission.tex
    with open("submission.tex", "r") as f:
        latex_content = f.read()

    # Generate the table content
    new_table_rows = []
    
    # Clean section
    new_table_rows.append(f"Clean & Static & {data['Static']['clean']['MNIST_Acc']:.2f}\\% & {data['Static']['clean']['KMNIST_Acc']:.2f}\\% & {data['Static']['clean']['FashionMNIST_Acc']:.2f}\\% & {data['Static']['clean']['Overall_Acc']:.2f}\\% & {data['Static']['clean']['NDR']:.2f}\\% & {data['Static']['clean']['FPR']:.2f}\\% \\\\")
    new_table_rows.append(f"      & PROTO-TTMM & {data['PROTO-TTMM']['clean']['MNIST_Acc']:.2f}\\% & {data['PROTO-TTMM']['clean']['KMNIST_Acc']:.2f}\\% & {data['PROTO-TTMM']['clean']['FashionMNIST_Acc']:.2f}\\% & {data['PROTO-TTMM']['clean']['Overall_Acc']:.2f}\\% & {data['PROTO-TTMM']['clean']['NDR']:.2f}\\% & {data['PROTO-TTMM']['clean']['FPR']:.2f}\\% \\\\")
    new_table_rows.append(f"      & IGGS-OW & {data['IGGS-OW']['clean']['MNIST_Acc']:.2f}\\% & {data['IGGS-OW']['clean']['KMNIST_Acc']:.2f}\\% & {data['IGGS-OW']['clean']['FashionMNIST_Acc']:.2f}\\% & {data['IGGS-OW']['clean']['Overall_Acc']:.2f}\\% & {data['IGGS-OW']['clean']['NDR']:.2f}\\% & {data['IGGS-OW']['clean']['FPR']:.2f}\\% \\\\")
    new_table_rows.append(f"      & \\textbf{{Rob-OW (Ours)}} & \\textbf{{{data['Rob-OW']['clean']['MNIST_Acc']:.2f}\\%}} & \\textbf{{{data['Rob-OW']['clean']['KMNIST_Acc']:.2f}\\%}} & {data['Rob-OW']['clean']['FashionMNIST_Acc']:.2f}\\% & \\textbf{{{data['Rob-OW']['clean']['Overall_Acc']:.2f}\\%}} & {data['Rob-OW']['clean']['NDR']:.2f}\\% & {data['Rob-OW']['clean']['FPR']:.2f}\\% \\\\")
    
    new_table_rows.append("      \\midrule")
    
    # Corrupted section
    new_table_rows.append(f"Corrupted & Static & {data['Static']['corrupted']['MNIST_Acc']:.2f}\\% & {data['Static']['corrupted']['KMNIST_Acc']:.2f}\\% & {data['Static']['corrupted']['FashionMNIST_Acc']:.2f}\\% & {data['Static']['corrupted']['Overall_Acc']:.2f}\\% & {data['Static']['corrupted']['NDR']:.2f}\\% & {data['Static']['corrupted']['FPR']:.2f}\\% \\\\")
    new_table_rows.append(f"          & PROTO-TTMM & {data['PROTO-TTMM']['corrupted']['MNIST_Acc']:.2f}\\% & {data['PROTO-TTMM']['corrupted']['KMNIST_Acc']:.2f}\\% & {data['PROTO-TTMM']['corrupted']['FashionMNIST_Acc']:.2f}\\% & {data['PROTO-TTMM']['corrupted']['Overall_Acc']:.2f}\\% & {data['PROTO-TTMM']['corrupted']['NDR']:.2f}\\% & {data['PROTO-TTMM']['corrupted']['FPR']:.2f}\\% \\\\")
    new_table_rows.append(f"          & IGGS-OW & {data['IGGS-OW']['corrupted']['MNIST_Acc']:.2f}\\% & {data['IGGS-OW']['corrupted']['KMNIST_Acc']:.2f}\\% & {data['IGGS-OW']['corrupted']['FashionMNIST_Acc']:.2f}\\% & {data['IGGS-OW']['corrupted']['Overall_Acc']:.2f}\\% & {data['IGGS-OW']['corrupted']['NDR']:.2f}\\% & {data['IGGS-OW']['corrupted']['FPR']:.2f}\\% \\\\")
    new_table_rows.append(f"          & \\textbf{{Rob-OW (Ours)}} & \\textbf{{{data['Rob-OW']['corrupted']['MNIST_Acc']:.2f}\\%}} & \\textbf{{{data['Rob-OW']['corrupted']['KMNIST_Acc']:.2f}\\%}} & {data['Rob-OW']['corrupted']['FashionMNIST_Acc']:.2f}\\% & \\textbf{{{data['Rob-OW']['corrupted']['Overall_Acc']:.2f}\\%}} & {data['Rob-OW']['corrupted']['NDR']:.2f}\\% & {data['Rob-OW']['corrupted']['FPR']:.2f}\\% \\\\")

    rows_str = "\n".join(new_table_rows)

    # Simple string slicing to avoid any regex escaping problems
    start_marker = "\\midrule"
    end_marker = "\\bottomrule"
    
    # We want to find the first occurrence of \midrule that is followed by the table content
    # Let's search from the start of the table environment
    table_idx = latex_content.find("\\label{tab:main_results}")
    if table_idx == -1:
        print("Error: Could not find label{tab:main_results}")
        return
        
    start_idx = latex_content.find(start_marker, table_idx)
    if start_idx == -1:
        print("Error: Could not find \\midrule after label")
        return
        
    end_idx = latex_content.find(end_marker, start_idx)
    if end_idx == -1:
        print("Error: Could not find \\bottomrule after \\midrule")
        return
        
    # Reconstruct the string
    # start_idx + len(start_marker) is right after \midrule
    updated_latex = (
        latex_content[:start_idx + len(start_marker)] +
        "\n" + rows_str + "\n" +
        latex_content[end_idx:]
    )
    
    with open("submission.tex", "w") as f:
        f.write(updated_latex)
    print("submission.tex updated successfully with correct table rows via slicing!")

if __name__ == "__main__":
    main()
