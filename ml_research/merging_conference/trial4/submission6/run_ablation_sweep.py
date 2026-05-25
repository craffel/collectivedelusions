import torch
from test_tta import evaluate_tta

if __name__ == "__main__":
    print("\n=======================================================")
    print("RUNNING HYPERPARAMETER ABLATION SWEEPS FOR CPA-MERGE")
    print("=======================================================")

    # Define sweep values
    tau_vals = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
    mask_vals = [0.0, 0.5, 0.7, 0.85, 0.95]
    beta_vals = [0.0, 0.05, 0.1, 0.5, 1.0]

    # Standard settings for other parameters
    default_tau = 0.02
    default_mask = 0.85
    default_beta = 0.1

    print("\n--- Sweeping Routing Temperature (tau) ---")
    tau_results = {}
    for tau in tau_vals:
        acc_clean = evaluate_tta("cpa_merge", "sequential", "clean", tau=tau, beta=default_beta, mask_threshold=default_mask)
        acc_noise = evaluate_tta("cpa_merge", "sequential", "noise", tau=tau, beta=default_beta, mask_threshold=default_mask)
        tau_results[tau] = {"clean": acc_clean, "noise": acc_noise}
        print(f"tau = {tau:.3f} | Clean Acc: {acc_clean:.2f}% | Noise Acc: {acc_noise:.2f}%")

    print("\n--- Sweeping Confidence Mask Threshold ---")
    mask_results = {}
    for mask in mask_vals:
        acc_clean = evaluate_tta("cpa_merge", "sequential", "clean", tau=default_tau, beta=default_beta, mask_threshold=mask)
        acc_noise = evaluate_tta("cpa_merge", "sequential", "noise", tau=default_tau, beta=default_beta, mask_threshold=mask)
        mask_results[mask] = {"clean": acc_clean, "noise": acc_noise}
        print(f"mask = {mask:.2f} | Clean Acc: {acc_clean:.2f}% | Noise Acc: {acc_noise:.2f}%")

    print("\n--- Sweeping Contrastive Weight (beta) ---")
    beta_results = {}
    for beta in beta_vals:
        acc_clean = evaluate_tta("cpa_merge", "sequential", "clean", tau=default_tau, beta=beta, mask_threshold=default_mask)
        acc_noise = evaluate_tta("cpa_merge", "sequential", "noise", tau=default_tau, beta=beta, mask_threshold=default_mask)
        beta_results[beta] = {"clean": acc_clean, "noise": acc_noise}
        print(f"beta = {beta:.2f} | Clean Acc: {acc_clean:.2f}% | Noise Acc: {acc_noise:.2f}%")

    # Generate Markdown and LaTeX outputs
    print("\n=======================================================")
    print("SWEEP RESULTS - LATEX TABLES")
    print("=======================================================")

    # Tau Table
    print("\n% --- Tau Sweep Table ---")
    print("\\begin{table}[h]")
    print("\\caption{Sensitivity analysis of the routing temperature $\\tau$ on sequential streams.}")
    print("\\label{tab:tau-sweep}")
    print("\\vskip 0.1in")
    print("\\begin{center}")
    print("\\begin{small}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Temperature ($\\tau$) & Clean Accuracy & Noise Accuracy \\\\")
    print("\\midrule")
    for tau in tau_vals:
        bold_prefix = "\\mathbf{" if tau == default_tau else ""
        bold_suffix = "}" if tau == default_tau else ""
        print(f"{bold_prefix}{tau}{bold_suffix} & {tau_results[tau]['clean']:.2f}\\% & {tau_results[tau]['noise']:.2f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{small}")
    print("\\end{center}")
    print("\\end{table}")

    # Mask Table
    print("\n% --- Mask Sweep Table ---")
    print("\\begin{table}[h]")
    print("\\caption{Sensitivity analysis of the confidence mask threshold on sequential streams.}")
    print("\\label{tab:mask-sweep}")
    print("\\vskip 0.1in")
    print("\\begin{center}")
    print("\\begin{small}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Threshold & Clean Accuracy & Noise Accuracy \\\\")
    print("\\midrule")
    for mask in mask_vals:
        bold_prefix = "\\mathbf{" if mask == default_mask else ""
        bold_suffix = "}" if mask == default_mask else ""
        print(f"{bold_prefix}{mask}{bold_suffix} & {mask_results[mask]['clean']:.2f}\\% & {mask_results[mask]['noise']:.2f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{small}")
    print("\\end{center}")
    print("\\end{table}")

    # Beta Table
    print("\n% --- Beta Sweep Table ---")
    print("\\begin{table}[h]")
    print("\\caption{Sensitivity analysis of the contrastive loss weight $\\beta$ on sequential streams.}")
    print("\\label{tab:beta-sweep}")
    print("\\vskip 0.1in")
    print("\\begin{center}")
    print("\\begin{small}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Weight ($\\beta$) & Clean Accuracy & Noise Accuracy \\\\")
    print("\\midrule")
    for beta in beta_vals:
        bold_prefix = "\\mathbf{" if beta == default_beta else ""
        bold_suffix = "}" if beta == default_beta else ""
        print(f"{bold_prefix}{beta}{bold_suffix} & {beta_results[beta]['clean']:.2f}\\% & {beta_results[beta]['noise']:.2f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{small}")
    print("\\end{center}")
    print("\\end{table}")

    # Write results to a summary file
    with open("ablation_results.md", "w") as f:
        f.write("# Hyperparameter Ablation Results\n\n")
        f.write("## 1. Routing Temperature (tau)\n\n")
        f.write("| tau | Clean Acc | Noise Acc |\n|---|---|---|\n")
        for tau in tau_vals:
            f.write(f"| {tau} | {tau_results[tau]['clean']:.2f}% | {tau_results[tau]['noise']:.2f}% |\n")
        
        f.write("\n## 2. Confidence Mask Threshold\n\n")
        f.write("| Threshold | Clean Acc | Noise Acc |\n|---|---|---|\n")
        for mask in mask_vals:
            f.write(f"| {mask} | {mask_results[mask]['clean']:.2f}% | {mask_results[mask]['noise']:.2f}% |\n")
            
        f.write("\n## 3. Contrastive Loss Weight (beta)\n\n")
        f.write("| beta | Clean Acc | Noise Acc |\n|---|---|---|\n")
        for beta in beta_vals:
            f.write(f"| {beta} | {beta_results[beta]['clean']:.2f}% | {beta_results[beta]['noise']:.2f}% |\n")

    print("\nAll sweeps complete and results saved to ablation_results.md!")
