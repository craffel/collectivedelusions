import os
import json
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)

def main():
    if not os.path.exists("results.json"):
        print("results.json not found. Run evaluation first.")
        return
        
    with open("results.json", "r") as f:
        data = json.load(f)
        
    runs = data["runs"]
    oracle_avg = data["oracle_avg"]
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14})
    
    # Plot 1: Clipping Curves (QCOT vs QWC) under INT4 Per-Channel PTQ with BN=32, Clean
    qcot_clean_int4 = [r for r in runs if "QCOT" in r["method"] and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "clean"]
    qwc_clean_int4 = [r for r in runs if "QWC" in r["method"] and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "clean"]
    
    # Baseline runs
    wa_clean_int4 = [r for r in runs if r["method"] == "WA" and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "clean"][0]
    ta_clean_int4 = [r for r in runs if r["method"] == "TA (lambda=0.4)" and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "clean"][0]
    
    # Parse parameter values and accuracies
    qcot_pairs = []
    for r in qcot_clean_int4:
        c_val = float(r["method"].split("=")[1][:-1])
        qcot_pairs.append((c_val, r["average"]))
    qcot_pairs.sort()
    
    qwc_pairs = []
    for r in qwc_clean_int4:
        q_val = float(r["method"].split("=")[1][:-1])
        qwc_pairs.append((q_val, r["average"]))
    qwc_pairs.sort()
    
    # Plot 1a: QCOT Clipping Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    c_vals, qcot_accs = zip(*qcot_pairs)
    ax1.plot(c_vals, qcot_accs, 'o-', color='tab:blue', linewidth=2, label='QCOT')
    ax1.axhline(y=wa_clean_int4["average"], color='tab:red', linestyle='--', label='WA (No Clipping)')
    ax1.axhline(y=ta_clean_int4["average"], color='tab:green', linestyle='--', label='TA (No Clipping)')
    ax1.set_xlabel('Clipping Threshold (C)')
    ax1.set_ylabel('Average Multitask Accuracy (%)')
    ax1.set_title('QCOT: Accuracy vs. Clipping Threshold (C)\n(INT4 Per-Channel, BN=32)')
    ax1.legend()
    
    # Plot 1b: QWC Clipping Curve (Quantile)
    q_vals, qwc_accs = zip(*qwc_pairs)
    ax2.plot(q_vals, qwc_accs, 's-', color='tab:purple', linewidth=2, label='QWC (Ours)')
    ax2.axhline(y=wa_clean_int4["average"], color='tab:red', linestyle='--')
    ax2.axhline(y=ta_clean_int4["average"], color='tab:green', linestyle='--')
    ax2.set_xlabel('Quantile Threshold (q)')
    ax2.set_ylabel('Average Multitask Accuracy (%)')
    ax2.set_title('QWC (Ours): Accuracy vs. Quantile (q)\n(INT4 Per-Channel, BN=32)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("results/clipping_curves.png", dpi=300)
    plt.close()
    
    # Plot 2: BatchNorm Healing Effect (Clean vs Noise vs Blur)
    best_qcot_acc = max(qcot_accs)
    best_qcot_method = [r["method"] for r in qcot_clean_int4 if r["average"] == best_qcot_acc][0]
    
    best_qwc_acc = max(qwc_accs)
    best_qwc_method = [r["method"] for r in qwc_clean_int4 if r["average"] == best_qwc_acc][0]
    
    environments = ["clean", "noise", "blur"]
    methods_to_compare = ["WA", "TA (lambda=0.4)", best_qcot_method, best_qwc_method, "EMQC", "CWSS", "CWSS-QC (q=0.9999)"]
    method_labels = ["WA", "TA", "QCOT (Best)", "QWC (Ours, Best)", "EMQC (Ours, Adaptive)", "CWSS (Ours)", "CWSS-QC (Ours, Best)"]
    
    bn0_accs = {m: [] for m in methods_to_compare}
    bn32_accs = {m: [] for m in methods_to_compare}
    
    for m in methods_to_compare:
        for env in environments:
            r0 = [r for r in runs if r["method"] == m and r["precision"] == "INT4_Channel" and r["bn_calib"] == 0 and r["corruption"] == env][0]
            r32 = [r for r in runs if r["method"] == m and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == env][0]
            bn0_accs[m].append(r0["average"])
            bn32_accs[m].append(r32["average"])
            
    # Bar Chart
    x = np.arange(len(environments))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # BN = 0
    multiplier = 0
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 'tab:cyan', 'tab:brown']
    for (m, label), color in zip(zip(methods_to_compare, method_labels), colors):
        offset = width * multiplier / len(methods_to_compare)
        ax1.bar(x + offset, bn0_accs[m], width/len(methods_to_compare), label=label, color=color)
        multiplier += 1
    ax1.set_xlabel('Environment')
    ax1.set_ylabel('Average Multitask Accuracy (%)')
    ax1.set_title('Without BatchNorm Calibration (BN=0)')
    ax1.set_xticks(x + width/2 - width/(2*len(methods_to_compare)))
    ax1.set_xticklabels([e.capitalize() for e in environments])
    ax1.set_ylim(0, 100)
    ax1.legend()
    
    # BN = 32
    multiplier = 0
    for (m, label), color in zip(zip(methods_to_compare, method_labels), colors):
        offset = width * multiplier / len(methods_to_compare)
        ax2.bar(x + offset, bn32_accs[m], width/len(methods_to_compare), label=label, color=color)
        multiplier += 1
    ax2.set_xlabel('Environment')
    ax2.set_title('With Data-Efficient BN Calibration (BN=32)')
    ax2.set_xticks(x + width/2 - width/(2*len(methods_to_compare)))
    ax2.set_xticklabels([e.capitalize() for e in environments])
    ax2.set_ylim(0, 100)
    ax2.legend()
    
    plt.suptitle('BatchNorm Calibration (DE-BN) Healing Effect under INT4 PTQ', fontsize=16)
    plt.tight_layout()
    plt.savefig("results/de_bn_healing.png", dpi=300)
    plt.close()
    
    # Generate LaTeX table of main results
    print("Generating LaTeX table...")
    latex_content = r"""\begin{table*}[t]
\centering
\caption{Comprehensive evaluation of model merging and calibration algorithms across MNIST, Fashion-MNIST, and CIFAR-10 tasks using a ResNet-18 backbone. Accuracy (\%) is reported under different precision regimes (FP32, 8-bit, 4-bit) and environmental corruptions (Clean, Gaussian Noise, Gaussian Blur) with task-specific Data-Efficient BatchNorm Calibration ($N=32$ samples).}
\label{tab:main_results}
\small
\setlength{\tabcolsep}{4.0pt}
\begin{tabular}{l c cccc cc}
\hline
\textbf{Method} & \textbf{Calibration} & \multicolumn{4}{c}{\textbf{Clean Accuracy (\%)}} & \textbf{Noise} & \textbf{Blur} \\
\cline{3-6}
& & \textbf{FP32} & \textbf{INT8-Tensor} & \textbf{INT8-Channel} & \textbf{INT4-Channel} & \textbf{Average} & \textbf{Average} \\
\hline
Oracle (Experts) & None & 90.27 & 90.27 & 90.27 & 90.27 & - & - \\
\hline
"""
    
    for m, m_lbl in zip(methods_to_compare, method_labels):
        fp32_clean = [r["average"] for r in runs if r["method"] == m and r["precision"] == "FP32" and r["bn_calib"] == 32 and r["corruption"] == "clean"][0]
        int8_t_clean = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT8_Tensor" and r["bn_calib"] == 32 and r["corruption"] == "clean"][0]
        int8_c_clean = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT8_Channel" and r["bn_calib"] == 32 and r["corruption"] == "clean"][0]
        int4_c_clean = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "clean"][0]
        noise_avg = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "noise"][0]
        blur_avg = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT4_Channel" and r["bn_calib"] == 32 and r["corruption"] == "blur"][0]
        
        latex_content += f"{m_lbl} & DE-BN ($N=32$) & {fp32_clean:.2f} & {int8_t_clean:.2f} & {int8_c_clean:.2f} & {int4_c_clean:.2f} & {noise_avg:.2f} & {blur_avg:.2f} \\\\\n"
        
    latex_content += "\\hline\n"
    latex_content += "\\multicolumn{8}{l}{\\textit{Uncalibrated Baselines (Without DE-BN calibration, $N=0$)}}\\\\\n"
    
    for m, m_lbl in zip(methods_to_compare, method_labels):
        fp32_clean = [r["average"] for r in runs if r["method"] == m and r["precision"] == "FP32" and r["bn_calib"] == 0 and r["corruption"] == "clean"][0]
        int8_t_clean = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT8_Tensor" and r["bn_calib"] == 0 and r["corruption"] == "clean"][0]
        int8_c_clean = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT8_Channel" and r["bn_calib"] == 0 and r["corruption"] == "clean"][0]
        int4_c_clean = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT4_Channel" and r["bn_calib"] == 0 and r["corruption"] == "clean"][0]
        noise_avg = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT4_Channel" and r["bn_calib"] == 0 and r["corruption"] == "noise"][0]
        blur_avg = [r["average"] for r in runs if r["method"] == m and r["precision"] == "INT4_Channel" and r["bn_calib"] == 0 and r["corruption"] == "blur"][0]
        
        latex_content += f"{m_lbl} & None ($N=0$) & {fp32_clean:.2f} & {int8_t_clean:.2f} & {int8_c_clean:.2f} & {int4_c_clean:.2f} & {noise_avg:.2f} & {blur_avg:.2f} \\\\\n"
        
    latex_content += r"""\hline
\end{tabular}
\end{table*}
"""
    
    with open("results/latex_table.tex", "w") as f:
        f.write(latex_content)
    print("LaTeX table generated successfully at results/latex_table.tex.")

if __name__ == "__main__":
    main()
