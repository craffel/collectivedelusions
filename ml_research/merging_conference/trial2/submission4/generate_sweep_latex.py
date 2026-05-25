import json

def format_latex_rho():
    with open("sweep_results_rho.json", "r") as f:
        data = json.load(f)
    print("% Table for SAM Radius sweep")
    print("\\begin{table}[h]")
    print("  \\caption{Ablation of SAM perturbation radius $\\rho$ (with fixed $lr = 0.02$, $\\beta = 0.0$). $\\rho = 0.0$ corresponds to Standard TTA.}")
    print("  \\label{table-ablation-rho}")
    print("  \\vskip 0.15in")
    print("  \\begin{center}")
    print("    \\begin{small}")
    print("      \\begin{tabular}{lccc}")
    print("        \\toprule")
    print("        SAM Radius $\\rho$ & CIFAR-10 Accuracy & SVHN Accuracy & Multi-Task Average \\\\")
    print("        \\midrule")
    for row in data:
        rho_str = f"{row['rho']:.2f}" if row['rho'] > 0 else "0.0 (Standard TTA)"
        print(f"        {rho_str:<17} & {row['cifar10']:.2f}\\% & {row['svhn']:.2f}\\% & {row['avg']:.2f}\\% \\\\")
    print("        \\bottomrule")
    print("      \\end{tabular}")
    print("    \\end{small}")
    print("  \\end{center}")
    print("  \\vskip -0.1in")
    print("\\end{table}")
    print()

def format_latex_lr():
    with open("sweep_results_lr.json", "r") as f:
        data = json.load(f)
    print("% Table for Learning Rate sweep")
    print("\\begin{table}[h]")
    print("  \\caption{Ablation of TTA learning rate $lr$ (with fixed $\\rho = 0.05$, $\\beta = 0.0$).}")
    print("  \\label{table-ablation-lr}")
    print("  \\vskip 0.15in")
    print("  \\begin{center}")
    print("    \\begin{small}")
    print("      \\begin{tabular}{lccc}")
    print("        \\toprule")
    print("        TTA Learning Rate $lr$ & CIFAR-10 Accuracy & SVHN Accuracy & Multi-Task Average \\\\")
    print("        \\midrule")
    for row in data:
        print(f"        {row['lr']:<22} & {row['cifar10']:.2f}\\% & {row['svhn']:.2f}\\% & {row['avg']:.2f}\\% \\\\")
    print("        \\bottomrule")
    print("      \\end{tabular}")
    print("    \\end{small}")
    print("  \\end{center}")
    print("  \\vskip -0.1in")
    print("\\end{table}")
    print()

def format_latex_beta():
    with open("sweep_results_beta.json", "r") as f:
        data = json.load(f)
    print("% Table for SOSR Weight sweep")
    print("\\begin{table}[h]")
    print("  \\caption{Ablation of SOSR regularization weight $\\beta$ (with fixed $\\rho = 0.05$, $lr = 0.02$). $\\beta = 0.0$ corresponds to SAM-Only TTA.}")
    print("  \\label{table-ablation-beta}")
    print("  \\vskip 0.15in")
    print("  \\begin{center}")
    print("    \\begin{small}")
    print("      \\begin{tabular}{lccc}")
    print("        \\toprule")
    print("        SOSR Weight $\\beta$ & CIFAR-10 Accuracy & SVHN Accuracy & Multi-Task Average \\\\")
    print("        \\midrule")
    for row in data:
        beta_str = f"{row['beta']:.2f}" if row['beta'] > 0 else "0.0 (SAM-Only TTA)"
        print(f"        {beta_str:<18} & {row['cifar10']:.2f}\\% & {row['svhn']:.2f}\\% & {row['avg']:.2f}\\% \\\\")
    print("        \\bottomrule")
    print("      \\end{tabular}")
    print("    \\end{small}")
    print("  \\end{center}")
    print("  \\vskip -0.1in")
    print("\\end{table}")
    print()

def main():
    format_latex_rho()
    format_latex_lr()
    format_latex_beta()

if __name__ == "__main__":
    main()
