import re

TEX_FILE = "submission.tex"

# Exact parsed values from log
main_data = [
    # N, Method, MNIST, F-MNIST, CIFAR, Mean, Mean+SFT
    ("4", "Uncalibrated (WA)", "58.21", "21.30", "23.00", "34.17", "23.97 $\\pm$ 3.06"),
    ("4", "SP-TAAC", "14.48 $\\pm$ 1.27", "45.60 $\\pm$ 2.92", "22.38 $\\pm$ 0.96", "27.49 $\\pm$ 1.70", "22.50 $\\pm$ 1.41"),
    ("4", "N-TAAC", "10.13 $\\pm$ 1.88", "10.11 $\\pm$ 2.17", "10.11 $\\pm$ 0.68", "10.12 $\\pm$ 0.96", "10.22 $\\pm$ 0.74"),
    ("4", "R-TAAC", "12.75 $\\pm$ 3.87", "10.44 $\\pm$ 0.74", "10.92 $\\pm$ 0.79", "11.37 $\\pm$ 1.34", "14.37 $\\pm$ 3.25"),
    ("4", "\\textbf{HSC (Ours)}", "\\textbf{27.25} $\\pm$ \\textbf{5.30}", "\\textbf{34.45} $\\pm$ \\textbf{1.28}", "\\textbf{24.33} $\\pm$ \\textbf{0.51}", "\\textbf{28.68} $\\pm$ \\textbf{2.24}", "\\textbf{22.20} $\\pm$ \\textbf{2.20}"),
    
    ("16", "Uncalibrated (WA)", "58.21", "21.30", "23.00", "34.17", "43.64 $\\pm$ 3.39"),
    ("16", "SP-TAAC", "14.92 $\\pm$ 0.92", "46.41 $\\pm$ 1.19", "22.60 $\\pm$ 0.73", "27.98 $\\pm$ 0.93", "32.76 $\\pm$ 1.66"),
    ("16", "N-TAAC", "22.36 $\\pm$ 1.23", "22.13 $\\pm$ 1.71", "13.07 $\\pm$ 0.71", "19.19 $\\pm$ 1.04", "16.88 $\\pm$ 2.68"),
    ("16", "R-TAAC", "12.94 $\\pm$ 1.43", "13.06 $\\pm$ 0.35", "11.34 $\\pm$ 1.51", "12.45 $\\pm$ 0.78", "22.15 $\\pm$ 3.60"),
    ("16", "\\textbf{HSC (Ours)}", "\\textbf{27.69} $\\pm$ \\textbf{1.11}", "\\textbf{39.88} $\\pm$ \\textbf{2.27}", "\\textbf{26.71} $\\pm$ \\textbf{0.65}", "\\textbf{31.43} $\\pm$ \\textbf{0.49}", "\\textbf{33.87} $\\pm$ \\textbf{2.42}"),
    
    ("64", "Uncalibrated (WA)", "58.21", "21.30", "23.00", "34.17", "60.14 $\\pm$ 0.44"),
    ("64", "SP-TAAC", "14.58 $\\pm$ 0.30", "45.63 $\\pm$ 0.47", "22.02 $\\pm$ 0.27", "27.41 $\\pm$ 0.33", "47.56 $\\pm$ 1.16"),
    ("64", "N-TAAC", "33.66 $\\pm$ 1.63", "24.56 $\\pm$ 1.56", "18.12 $\\pm$ 1.25", "25.45 $\\pm$ 0.86", "28.68 $\\pm$ 1.49"),
    ("64", "R-TAAC", "11.33 $\\pm$ 3.13", "12.91 $\\pm$ 1.13", "10.96 $\\pm$ 0.23", "11.73 $\\pm$ 1.14", "35.63 $\\pm$ 0.80"),
    ("64", "\\textbf{HSC (Ours)}", "\\textbf{31.40} $\\pm$ \\textbf{2.13}", "\\textbf{41.12} $\\pm$ \\textbf{0.39}", "\\textbf{28.00} $\\pm$ \\textbf{1.10}", "\\textbf{33.51} $\\pm$ \\textbf{0.55}", "\\textbf{46.83} $\\pm$ \\textbf{0.55}"),
    
    ("128", "Uncalibrated (WA)", "58.21", "21.30", "23.00", "34.17", "66.15 $\\pm$ 2.61"),
    ("128", "SP-TAAC", "15.06 $\\pm$ 0.46", "46.55 $\\pm$ 1.41", "22.28 $\\pm$ 0.39", "27.96 $\\pm$ 0.74", "53.63 $\\pm$ 1.29"),
    ("128", "N-TAAC", "33.53 $\\pm$ 0.24", "24.29 $\\pm$ 0.43", "19.23 $\\pm$ 1.22", "25.68 $\\pm$ 0.60", "32.28 $\\pm$ 0.29"),
    ("128", "R-TAAC", "13.36 $\\pm$ 3.50", "11.83 $\\pm$ 1.68", "10.02 $\\pm$ 0.81", "11.74 $\\pm$ 0.38", "45.62 $\\pm$ 1.72"),
    ("128", "\\textbf{HSC (Ours)}", "\\textbf{32.87} $\\pm$ \\textbf{2.63}", "\\textbf{39.38} $\\pm$ \\textbf{0.71}", "\\textbf{28.32} $\\pm$ \\textbf{0.35}", "\\textbf{33.52} $\\pm$ \\textbf{1.17}", "\\textbf{52.70} $\\pm$ \\textbf{0.99}"),
    
    ("256", "Uncalibrated (WA)", "58.21", "21.30", "23.00", "34.17", "69.82 $\\pm$ 0.29"),
    ("256", "SP-TAAC", "15.15 $\\pm$ 0.24", "46.62 $\\pm$ 0.19", "22.65 $\\pm$ 0.28", "28.14 $\\pm$ 0.19", "58.34 $\\pm$ 0.71"),
    ("256", "N-TAAC", "33.91 $\\pm$ 1.38", "23.79 $\\pm$ 0.61", "19.30 $\\pm$ 0.97", "25.67 $\\pm$ 0.81", "36.15 $\\pm$ 0.59"),
    ("256", "R-TAAC", "14.74 $\\pm$ 3.31", "11.22 $\\pm$ 2.42", "11.41 $\\pm$ 0.50", "12.46 $\\pm$ 0.62", "47.81 $\\pm$ 2.44"),
    ("256", "\\textbf{HSC (Ours)}", "\\textbf{32.21} $\\pm$ \\textbf{1.78}", "\\textbf{42.27} $\\pm$ \\textbf{2.89}", "\\textbf{28.57} $\\pm$ \\textbf{0.47}", "\\textbf{34.35} $\\pm$ \\textbf{1.13}", "\\textbf{57.85} $\\pm$ \\textbf{0.58}"),
]

split_data = [
    ("0 (Pure R-TAAC)", "11.74 $\\pm$ 0.38", "45.62 $\\pm$ 1.72"),
    ("5", "16.63 $\\pm$ 1.00", "48.29 $\\pm$ 1.89"),
    ("10", "27.72 $\\pm$ 0.45", "51.70 $\\pm$ 1.33"),
    ("\\textbf{15 (Layer 4 Block)}", "\\textbf{33.52} $\\pm$ \\textbf{1.17}", "\\textbf{52.70} $\\pm$ \\textbf{0.99}"),
    ("19", "29.29 $\\pm$ 0.41", "52.23 $\\pm$ 1.05"),
]

alpha_data = [
    ("0.00", "32.46 $\\pm$ 1.56", "51.85 $\\pm$ 1.99"),
    ("\\textbf{0.25}", "\\textbf{35.42} $\\pm$ \\textbf{1.30}", "\\textbf{54.95} $\\pm$ \\textbf{1.22}"),
    ("0.50", "33.52 $\\pm$ 1.17", "52.70 $\\pm$ 0.99"),
    ("0.75", "31.80 $\\pm$ 0.60", "49.27 $\\pm$ 0.46"),
    ("1.00", "31.46 $\\pm$ 0.20", "43.17 $\\pm$ 1.59"),
]

# Generate Main Table
main_table_lines = [
    "\\begin{table*}[t]",
    "\\caption{Multi-task model merging accuracies (\\%) on ResNet-18 across calibration sizes $N$ (aggregated over multiple seeds). Methods are evaluated both without head SFT (Representation only) and with head SFT (Mean+SFT).}",
    "\\label{tab:main}",
    "\\vskip 0.15in",
    "\\begin{center}",
    "\\begin{small}",
    "\\begin{tabular}{llccccc}",
    "\\toprule",
    "N & Method & MNIST Acc & F-MNIST Acc & CIFAR-10 Acc & Mean Acc & Mean+SFT Acc \\\\",
    "\\midrule",
    "- & Expert Models (Upper Bound) & 98.45 & 85.40 & 67.38 & 83.74 & -- \\\\",
    "\\midrule"
]

prev_n = None
for row in main_data:
    n_val, method, mnist, fashion, cifar, mean, sft = row
    if prev_n is not None and prev_n != n_val:
        main_table_lines.append("\\midrule")
    main_table_lines.append(f"{n_val} & {method} & {mnist} & {fashion} & {cifar} & {mean} & {sft} \\\\")
    prev_n = n_val

main_table_lines.extend([
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{small}",
    "\\end{center}",
    "\\vskip -0.15in",
    "\\vspace{-3mm}",
    "\\end{table*}"
])
main_table_str = "\n".join(main_table_lines)


# Generate Combined Split and Alpha tables side-by-side
split_lines = [
    "\\begin{tabular}{ccc}",
    "\\toprule",
    "Split Index ($M$) & Mean Acc (\\%) & Mean+SFT Acc (\\%) \\\\",
    "\\midrule"
]
for row in split_data:
    split_lines.append(f"{row[0]} & {row[1]} & {row[2]} \\\\")
split_lines.append("\\bottomrule")
split_lines.append("\\end{tabular}")
split_tabular_str = "\n".join(split_lines)

alpha_lines = [
    "\\begin{tabular}{ccc}",
    "\\toprule",
    "Alpha ($\\alpha$) & Mean Acc (\\%) & Mean+SFT Acc (\\%) \\\\",
    "\\midrule"
]
for row in alpha_data:
    alpha_lines.append(f"{row[0]} & {row[1]} & {row[2]} \\\\")
alpha_lines.append("\\bottomrule")
alpha_lines.append("\\end{tabular}")
alpha_tabular_str = "\n".join(alpha_lines)


# Read submission.tex
with open(TEX_FILE, 'r') as f:
    tex = f.read()

# Replace main table
tex_replaced = re.sub(
    r"\\begin\{table\*\}\[t\].*?\\label\{tab:main\}.*?\\end\{table\*\}",
    main_table_str.replace("\\", "\\\\"),
    tex,
    flags=re.DOTALL
)

# Surgically replace split minipage tabular
split_minipage_pattern = r"(\\begin\{minipage\}\{0\.48\\textwidth\}\s*\\centering\s*\\caption\{.*?split.*?\}.*?\\label\{tab:split\}.*?\\begin\{tabular\}.*?\\end\{tabular\}\s*\\end\{small\}\s*\\end\{minipage\})"
split_match = re.search(split_minipage_pattern, tex_replaced, re.DOTALL)
if split_match:
    old_split_mp = split_match.group(1)
    new_split_mp = re.sub(
        r"\\begin\{tabular\}.*?\\end\{tabular\}",
        split_tabular_str.replace("\\", "\\\\"),
        old_split_mp,
        flags=re.DOTALL
    )
    tex_replaced = tex_replaced.replace(old_split_mp, new_split_mp)

# Surgically replace alpha minipage tabular
alpha_minipage_pattern = r"(\\begin\{minipage\}\{0\.48\\textwidth\}\s*\\centering\s*\\caption\{.*?alpha.*?\}.*?\\label\{tab:alpha\}.*?\\begin\{tabular\}.*?\\end\{tabular\}\s*\\end\{small\}\s*\\end\{minipage\})"
alpha_match = re.search(alpha_minipage_pattern, tex_replaced, re.DOTALL)
if alpha_match:
    old_alpha_mp = alpha_match.group(1)
    new_alpha_mp = re.sub(
        r"\\begin\{tabular\}.*?\\end\{tabular\}",
        alpha_tabular_str.replace("\\", "\\\\"),
        old_alpha_mp,
        flags=re.DOTALL
    )
    tex_replaced = tex_replaced.replace(old_alpha_mp, new_alpha_mp)

# Save back
with open(TEX_FILE, 'w') as f:
    f.write(tex_replaced)

print("Tables fixed successfully with exact math symbols!")
