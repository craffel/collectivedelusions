import json
import numpy as np
import os

def process():
    seeds = [42, 43, 44]
    results_list = []
    for s in seeds:
        filename = f"results_seed{s}.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                results_list.append(json.load(f))
                
    if not results_list:
        print("No results files found yet.")
        return
        
    sc_ids = ["A", "B", "C", "D", "E"]
    sc_names = {
        "A": "SGD (Standard)",
        "B": "SGD (High Decay)",
        "C": "AdamW (Standard)",
        "D": "AdamW (High LR)",
        "E": "AdamW (High Decay)"
    }
    
    print("\n" + "="*50)
    print("PROCESSED MULTI-SEED RESULTS")
    print("="*50)
    
    # Process Geometries
    print("\n--- PARAMETER-SPACE GEOMETRIES ---")
    for sc_id in sc_ids:
        if sc_id not in results_list[0]:
            continue
        drifts_mnist = []
        drifts_fmnist = []
        drifts_cifar = []
        cos_sims_12 = []
        cos_sims_13 = []
        cos_sims_23 = []
        avg_cos_sims = []
        
        for res in results_list:
            sc_data = res[sc_id]
            drifts_mnist.append(sc_data["drifts"][0])
            drifts_fmnist.append(sc_data["drifts"][1])
            drifts_cifar.append(sc_data["drifts"][2])
            cos_sims_12.append(sc_data["cos_sims"][0])
            cos_sims_13.append(sc_data["cos_sims"][1])
            cos_sims_23.append(sc_data["cos_sims"][2])
            avg_cos_sims.append(sc_data["avg_cos_sim"])
            
        print(f"\nScenario {sc_id}: {sc_names[sc_id]}")
        print(f"  Drift MNIST:      {np.mean(drifts_mnist):.4f} +/- {np.std(drifts_mnist):.4f}")
        print(f"  Drift F-MNIST:    {np.mean(drifts_fmnist):.4f} +/- {np.std(drifts_fmnist):.4f}")
        print(f"  Drift CIFAR-10:   {np.mean(drifts_cifar):.4f} +/- {np.std(drifts_cifar):.4f}")
        print(f"  CosSim (M/FM):    {np.mean(cos_sims_12):.4f} +/- {np.std(cos_sims_12):.4f}")
        print(f"  CosSim (M/C10):   {np.mean(cos_sims_13):.4f} +/- {np.std(cos_sims_13):.4f}")
        print(f"  CosSim (FM/C10):  {np.mean(cos_sims_23):.4f} +/- {np.std(cos_sims_23):.4f}")
        print(f"  Avg CosSim:       {np.mean(avg_cos_sims):.4f} +/- {np.std(avg_cos_sims):.4f}")

    # Process Accuracies
    print("\n--- MODEL ACCURACIES (AVERAGE OVER TASKS %) ---")
    for sc_id in sc_ids:
        if sc_id not in results_list[0]:
            continue
        uncal = []
        bnc = []
        sptaac = []
        hybrid4 = []
        hybrid8 = []
        
        for res in results_list:
            sc_data = res[sc_id]
            uncal.append(sc_data["avg_uncal"])
            bnc.append(sc_data["avg_bnc"])
            sptaac.append(sc_data["avg_sptaac"])
            hybrid4.append(sc_data["avg_hybrid4"])
            hybrid8.append(sc_data["avg_hybrid8"])
            
        print(f"\nScenario {sc_id}: {sc_names[sc_id]}")
        print(f"  Uncalibrated:       {np.mean(uncal):.2f}% +/- {np.std(uncal):.2f}%")
        print(f"  BNC Calibrated:     {np.mean(bnc):.2f}% +/- {np.std(bnc):.2f}%")
        print(f"  SP-TAAC Calibrated: {np.mean(sptaac):.2f}% +/- {np.std(sptaac):.2f}%")
        print(f"  Hybrid (Rank 4):    {np.mean(hybrid4):.2f}% +/- {np.std(hybrid4):.2f}%")
        print(f"  Hybrid (Rank 8):    {np.mean(hybrid8):.2f}% +/- {np.std(hybrid8):.2f}%")
        
    # Generate LaTeX code for the geometry table
    print("\n" + "="*50)
    print("LATEX GEOMETRY TABLE CODE")
    print("="*50)
    print(r"""\begin{table*}[t]
\caption{Expert Parameter Drift ($D(W_k)$) and Weight Update Cosine Alignment ($S(i,j)$) across Optimization Scenarios (Mean $\pm$ SD across 3 random seeds).}
\label{tab:geometries}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccccc}
\toprule
Metric & Scenario A (SGD Std) & Scenario B (SGD High Decay) & Scenario C (AdamW Std) & Scenario D (AdamW High LR) & Scenario E (AdamW High Decay) \\
\midrule""")
    
    # Fetch row values
    def get_latex_cells(metrics_list):
        cells = []
        for s_idx in sc_ids:
            if s_idx not in results_list[0]:
                cells.append("N/A")
                continue
            vals = [res[s_idx][metrics_list[0]][metrics_list[1]] if len(metrics_list)==2 else res[s_idx][metrics_list[0]] for res in results_list]
            cells.append(f"{np.mean(vals):.4f} $\\pm$ {np.std(vals):.4f}")
        return " & ".join(cells)
        
    print("MNIST Drift & " + get_latex_cells(["drifts", 0]) + r" \\")
    print("F-MNIST Drift & " + get_latex_cells(["drifts", 1]) + r" \\")
    print("CIFAR-10 Drift & " + get_latex_cells(["drifts", 2]) + r" \\")
    print("CosSim: M / FM & " + get_latex_cells(["cos_sims", 0]) + r" \\")
    print("CosSim: M / C10 & " + get_latex_cells(["cos_sims", 1]) + r" \\")
    print("CosSim: FM / C10 & " + get_latex_cells(["cos_sims", 2]) + r" \\")
    print("Avg CosSim & " + get_latex_cells(["avg_cos_sim"]) + r" \\")
    
    print(r"""\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table*}""")

    # Generate LaTeX code for the accuracy table
    print("\n" + "="*50)
    print("LATEX ACCURACY TABLE CODE")
    print("="*50)
    print(r"""\begin{table*}[t]
\caption{Multi-Task Merged Model Average Test Accuracies (\%) across Calibration Settings (Mean $\pm$ SD across 3 random seeds).}
\label{tab:accuracies}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccccc}
\toprule
Calibration Method & Scenario A & Scenario B & Scenario C & Scenario D & Scenario E \\
 & (SGD Std) & (SGD High Decay) & (AdamW Std) & (AdamW High LR) & (AdamW High Decay) \\
\midrule""")
    
    def get_acc_cells(key):
        cells = []
        for s_idx in sc_ids:
            if s_idx not in results_list[0]:
                cells.append("N/A")
                continue
            vals = [res[s_idx][key] for res in results_list]
            cells.append(f"{np.mean(vals):.2f}\\% $\\pm$ {np.std(vals):.2f}\\%")
        return " & ".join(cells)
        
    print("Uncalibrated (none) & " + get_acc_cells("avg_uncal") + r" \\")
    print("BNC Calibration & " + get_acc_cells("avg_bnc") + r" \\")
    print("SP-TAAC Calibration & " + get_acc_cells("avg_sptaac") + r" \\")
    print("Hybrid (Rank 4) & " + get_acc_cells("avg_hybrid4") + r" \\")
    print("Hybrid (Rank 8) & " + get_acc_cells("avg_hybrid8") + r" \\")
    
    print(r"""\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table*}""")

if __name__ == "__main__":
    process()
