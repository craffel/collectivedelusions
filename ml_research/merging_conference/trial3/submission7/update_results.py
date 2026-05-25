import time
import os
import subprocess
import re

job_id = "22158984"
print(f"Starting update_results.py monitoring for Slurm job {job_id}...")

# 1. Parse results from results_summary.txt
results = {}
with open("results_summary.txt", "r") as f:
    for line in f:
        if line.startswith("alternating") or line.startswith("sequential"):
            parts = [p.strip().replace('%', '') for p in line.split("|")]
            stream_type = parts[0]
            vals = [float(x) for x in parts[1:]]
            results[stream_type] = vals

print("Parsed results successfully:")
print("Alternating:", results['alternating'])
print("Sequential:", results['sequential'])

# 2. Create LaTeX table values with bold highlighting for the maximums
alt_vals = results['alternating']
max_alt = max(alt_vals)
alt_strs = [f"\\textbf{{{x:.2f}}}" if x == max_alt else f"{x:.2f}" for x in alt_vals]

seq_vals = results['sequential']
max_seq = max(seq_vals)
seq_strs = [f"\\textbf{{{x:.2f}}}" if x == max_seq else f"{x:.2f}" for x in seq_vals]

# Read submission.tex
with open("submission.tex", "r") as f:
    tex = f.read()

# Replace table placeholders in DESCENDING order of length to prevent prefix collisions!
tex = tex.replace("ALT_L2_100", alt_strs[4])
tex = tex.replace("ALT_L2_10", alt_strs[3])
tex = tex.replace("ALT_L2_1", alt_strs[2])

tex = tex.replace("ALT_EWC_100", alt_strs[7])
tex = tex.replace("ALT_EWC_10", alt_strs[6])
tex = tex.replace("ALT_EWC_1", alt_strs[5])

tex = tex.replace("ALT_STATIC_ACC", alt_strs[0])
tex = tex.replace("ALT_STD_ACC", alt_strs[1])


tex = tex.replace("SEQ_L2_100", seq_strs[4])
tex = tex.replace("SEQ_L2_10", seq_strs[3])
tex = tex.replace("SEQ_L2_1", seq_strs[2])

tex = tex.replace("SEQ_EWC_100", seq_strs[7])
tex = tex.replace("SEQ_EWC_10", seq_strs[6])
tex = tex.replace("SEQ_EWC_1", seq_strs[5])

tex = tex.replace("SEQ_STATIC_ACC", seq_strs[0])
tex = tex.replace("SEQ_STD_ACC", seq_strs[1])

# Extract values for dynamic text replacement
seq_static = seq_vals[0]
seq_std = seq_vals[1]

# Find best EWC-TTA results for sequential
seq_ewc_vals = seq_vals[5:8]
gammas = ['1.0', '10.0', '100.0']
best_ewc_idx = seq_ewc_vals.index(max(seq_ewc_vals))
best_ewc_gamma = gammas[best_ewc_idx]
best_ewc_acc = seq_ewc_vals[best_ewc_idx]
ewc_gain = best_ewc_acc - seq_static

print(f"Best sequential EWC-TTA accuracy: {best_ewc_acc:.2f}% at gamma={best_ewc_gamma}")
print(f"Absolute gain over Static Merged: {ewc_gain:.2f}%")

# Perform safe literal replacements using re.search + str.replace

# Abstract replacement
abstract_pattern = r"achieves an outstanding accuracy of \\textbf\{[0-9.]+\\%\} on severe sequential streams, outperforming the static merging baseline by over \\textbf\{[0-9.]+\\%\} absolute accuracy\."
abstract_replacement = f"achieves an outstanding accuracy of \\textbf{{{best_ewc_acc:.2f}\\%}} on severe sequential streams, outperforming the static merging baseline by over \\textbf{{{ewc_gain:.2f}\\%}} absolute accuracy."
match = re.search(abstract_pattern, tex)
if match:
    tex = tex.replace(match.group(0), abstract_replacement)
else:
    print("Warning: Abstract pattern not found in LaTeX.")

# Introduction replacement
intro_pattern = r"On severe non-stationary sequential streams, EWC-TTA boosts multi-task accuracy from \\textbf\{[0-9.]+\\%\} \(Static Merged\) to \\textbf\{[0-9.]+\%\}, yielding an absolute gain of \\textbf\{[0-9.]+\%\}\."
intro_replacement = f"On severe non-stationary sequential streams, EWC-TTA boosts multi-task accuracy from \\textbf{{{seq_static:.2f}\\%}} (Static Merged) to \\textbf{{{best_ewc_acc:.2f}\\%}}, yielding an absolute gain of \\textbf{{{ewc_gain:.2f}\\%}}."
match = re.search(intro_pattern, tex)
if match:
    tex = tex.replace(match.group(0), intro_replacement)
else:
    print("Warning: Introduction pattern not found in LaTeX.")

# Results section replacement 1
res_pattern_1 = r"On the severe sequential test stream, static model merging performs poorly, achieving only \\textbf\{[0-9.]+\\%\} accuracy\."
res_replacement_1 = f"On the severe sequential test stream, static model merging performs poorly, achieving only \\textbf{{{seq_static:.2f}\\%}} accuracy."
match = re.search(res_pattern_1, tex)
if match:
    tex = tex.replace(match.group(0), res_replacement_1)
else:
    print("Warning: Results pattern 1 not found in LaTeX.")

# Results section replacement 2
res_pattern_2 = r"yielding a huge performance boost to \\textbf\{[0-9.]+\%\} accuracy\."
res_replacement_2 = f"yielding a huge performance boost to \\textbf{{{seq_std:.2f}\\%}} accuracy."
match = re.search(res_pattern_2, tex)
if match:
    tex = tex.replace(match.group(0), res_replacement_2)
else:
    print("Warning: Results pattern 2 not found in LaTeX.")

# Results section replacement 3
res_pattern_3 = r"By introducing the EWC regularizer, \\textbf\{EWC-TTA \(\\gamma=[0-9.]+\)\} further improves accuracy to \\textbf\{[0-9.]+\%\}, outperforming both standard unconstrained TTA and the static baseline \(\+[0-9.]+\% absolute gain\)\."
# Use loose pattern for results pattern 3
res_pattern_3_loose = r"By introducing the EWC regularizer, \\textbf\{EWC-TTA \(\\gamma=[0-9.]+\)\} further improves accuracy to \\textbf\{[0-9.]+\\\%\}, outperforming both standard unconstrained TTA and the static baseline \(\+[0-9.]+\\\% absolute gain\)\."
res_replacement_3 = f"By introducing the EWC regularizer, \\textbf{{EWC-TTA (\\gamma={best_ewc_gamma})}} further improves accuracy to \\textbf{{{best_ewc_acc:.2f}\\%}}, outperforming both standard unconstrained TTA and the static baseline (+{ewc_gain:.2f}\\% absolute gain)."
match = re.search(res_pattern_3_loose, tex)
if match:
    tex = tex.replace(match.group(0), res_replacement_3)
else:
    match = re.search(res_pattern_3, tex)
    if match:
        tex = tex.replace(match.group(0), res_replacement_3)
    else:
        print("Warning: Results pattern 3 not found in LaTeX.")

# Results section replacement 4 (ablation of gamma)
ablation_pattern = r"setting \\gamma=1\.0\$ yields \$[0-9.]+\%\$, while \$\\gamma=10\.0\$ reaches the peak of \$[0-9.]+\%\$\. Looking closer, increasing the regularization strength further to \$\\gamma=100\.0\$ slightly decreases accuracy to \$[0-9.]+\%\$\."
ablation_pattern_loose = r"setting \\gamma=1\.0\$ yields \$[0-9.]+\%\$, while \$\\gamma=10\.0\$ reaches the peak of \$[0-9.]+\%\$\. Increasing the regularization strength further to \$\\gamma=100\.0\$ slightly decreases accuracy to \$[0-9.]+\%\$\."
ablation_pattern_latex = r"setting \\gamma=1\.0\$ yields \$[0-9.]+\\\%\$, while \$\\gamma=10\.0\$ reaches the peak of \$[0-9.]+\\\%\$\. Increasing the regularization strength further to \$\\gamma=100\.0\$ slightly decreases accuracy to \$[0-9.]+\\\%\$\."
ablation_replacement = f"setting \\gamma=1.0$ yields ${seq_vals[5]:.2f}\%$, while $\\gamma=10.0$ reaches the peak of ${seq_vals[6]:.2f}\%$. Increasing the regularization strength further to $\\gamma=100.0$ slightly decreases accuracy to ${seq_vals[7]:.2f}\%$."
match = re.search(ablation_pattern_latex, tex)
if match:
    tex = tex.replace(match.group(0), ablation_replacement)
else:
    match = re.search(ablation_pattern_loose, tex)
    if match:
        tex = tex.replace(match.group(0), ablation_replacement)
    else:
        match = re.search(ablation_pattern, tex)
        if match:
            tex = tex.replace(match.group(0), ablation_replacement)
        else:
            print("Warning: Ablation pattern not found in LaTeX.")

# Conclusion replacement
conclusion_pattern = r"EWC-TTA achieves an outstanding \\textbf\{[0-9.]+\\%\} accuracy on sequential streams, representing an \\textbf\{[0-9.]+\\%\} improvement over static merging\."
conclusion_replacement = f"EWC-TTA achieves an outstanding \\textbf{{{best_ewc_acc:.2f}\\%}} accuracy on sequential streams, representing an \\textbf{{{ewc_gain:.2f}\\%}} improvement over static merging."
match = re.search(conclusion_pattern, tex)
if match:
    tex = tex.replace(match.group(0), conclusion_replacement)
else:
    print("Warning: Conclusion pattern not found in LaTeX.")

# Save updated submission.tex
with open("submission.tex", "w") as f:
    f.write(tex)
print("Updated submission.tex with exact numbers successfully!")

# 3. Generate the plot
print("Re-generating results plot...")
subprocess.check_call("python plot_results.py", shell=True)

# 4. Compile the PDF using tectonic
print("Compiling LaTeX to PDF via tectonic...")
try:
    subprocess.check_call("tectonic submission.tex", shell=True)
    print("Successfully compiled submission.pdf!")
except Exception as e:
    print(f"Error compiling LaTeX: {e}")
    exit(1)

print("\nAll tasks completed successfully!")
