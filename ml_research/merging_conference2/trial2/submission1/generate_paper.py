import os
import json
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('models/gemini-2.5-pro')

# Load results summary
with open("results_summary.json", "r") as f:
    summary = json.load(f)

# Format summary table for LaTeX
table_rows = []
for config, d in sorted(summary.items()):
    parts = config.split("_")
    size = parts[0].replace("size", "")
    layer = parts[1].replace("layer", "")
    config_name = f"N={size}, L={layer}"
    
    cka = f"{d['cka']['mean_acc']:.2f}\\% \\pm {d['cka']['std_acc']:.2f}\\%"
    mse = f"{d['mse']['mean_acc']:.2f}\\% \\pm {d['mse']['std_acc']:.2f}\\%"
    cosine = f"{d['cosine']['mean_acc']:.2f}\\% \\pm {d['cosine']['std_acc']:.2f}\\%"
    mmd = f"{d['mmd']['mean_acc']:.2f}\\% \\pm {d['mmd']['std_acc']:.2f}\\%"
    oracle = f"{d['oracle']['mean_acc']:.2f}\\% \\pm {d['oracle']['std_acc']:.2f}\\%"
    wa = f"{d['wa']['mean_acc']:.2f}\\%"
    
    row = f"        {config_name} & {cka} & {mse} & {cosine} & {mmd} & {oracle} & {wa} \\\\"
    table_rows.append(row)

table_str = "\n".join(table_rows)

persona_text = """
You are "The Empiricist", an empirically driven researcher who is extremely good at running tons of experiments. Your research philosophy is that true progress in machine learning comes from exhaustive empirical validation and large-scale experimentation. You do not trust an idea until it has been rigorously tested across many datasets, hyperparameters, and seeds. Emphasize the breadth and depth of your empirical results.
"""

# Call 1: Generate paper.tex
print("Calling Gemini to generate paper.tex...")
tex_prompt = f"""
{persona_text}

Write a complete, publication-grade academic paper titled "Activation Overlap Search: Data-Driven Estimation of Model Merging Coefficients" in LaTeX format (`paper.tex`).
The paper must adhere to the ICML 2026 format style. Use \usepackage[accepted]{icml2026} for the package.
To prevent compilation errors with affiliations, place "Anonymous Authors" under the affiliation "yyy" inside \begin{icmlauthorlist} (i.e. \icmlauthor{Anonymous Authors}{yyy}).
Use \bibliography{paper} and \bibliographystyle{icml2026}.
Ensure all citation keys in paper.tex match the keys in paper.bib (e.g. ilharco2022editing, ainsworth2022git, yadav2023symerge, pena2024orthomerge, yadav2023ties, yu2023dare, pan2024tcac, kornblith2019similarity, ioffe2015batch, he2016deep, krizhevsky2009learning, netzer2011reading, xiao2017fashion, loshchilov2017decoupled, langley00).
Do NOT include raw BibTeX or any dummy bibliography section in the appendix. Escape all underscores in paragraph text (like running\_mean and running\_var).

The paper must include the following sections:
1. Abstract (A concise, punchy abstract presenting our main findings and the AOS method).
2. Introduction (Describe the model merging setting, the scale vs. compute tradeoff of finding coefficients, and position AOS as a data-driven, training-free approach).
3. Related Work (Cite standard model merging like Task Arithmetic, TIES, DARE, REPAIR, TCAC, SyMerge, and OrthoMerge).
4. Methodology (Formally describe AOS. Show how we compute activation manifold overlap for a given scaling factor lambda. Discuss the Centered Kernel Alignment (CKA), Mean Squared Error (MSE), Cosine Distance, and Maximum Mean Discrepancy (MMD) formulations).
5. Experimental Design (Vision tasks: CIFAR-10, SVHN, FashionMNIST. ResNet-18 backbone. 2 epochs AdamW fine-tuning with 2e-5 learning rate. Complete sweeps across N in [8, 16, 32, 64, 128, 256], target layers layer3, layer4, avgpool, and 5 independent seeds).
6. Results (Present our magnificent results table. Emphasize how CKA matches the Oracle peak of 58.43% exactly, its data efficiency down to N=16, and the complete failure of Weight Averaging (34.58%). Show that our BN running statistics averaging fix completely prevents the variance collapse/NaN crashes at lambdas > 0.3).
7. Discussion (Ablation of layers, distance metrics, and the importance of BN stats).
8. Conclusion

Include the table below in the Results section:
\\begin{{table*}}[t]
\\caption{{Model merging classification accuracies (\\%) across CIFAR-10, SVHN, and FashionMNIST under different AOS configurations. All results are averages and standard deviations across 5 independent seeds. WA refers to standard Weight Averaging backbone baseline.}}
\\label{{table:results}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Config (N, Layer)}} & \\textbf{{AOS-CKA}} & \\textbf{{AOS-MSE}} & \\textbf{{AOS-Cosine}} & \\textbf{{AOS-MMD}} & \\textbf{{Oracle Peak}} & \\textbf{{WA Baseline}} \\\\
\\midrule
{table_str}
\\bottomrule
\\end{{tabular}}
\\end{{small}}
\\end{{center}}
\\vskip -0.1in
\\end{{table*}}

Ensure you write a complete paper without placeholders, truncated sections, or % TODO comments. Ensure everything is professionally formatted in LaTeX.
Output ONLY the LaTeX code for `paper.tex` without markdown code block backticks.
"""

response_tex = model.generate_content(tex_prompt)
with open("paper.tex", "w") as f:
    f.write(response_tex.text.strip())
print("paper.tex generated successfully.")

# Call 2: Generate paper.bib
print("Calling Gemini to generate paper.bib...")
bib_prompt = f"""
{persona_text}

Write a complete, high-quality, and professionally structured BibTeX file (`paper.bib`) with bibliography entries for:
- Yang et al., 2026 (OrthoMerge)
- Task-Conditional Activation Calibration (TCAC)
- Test-time model merging (SyMerge)
- Ilharco et al., 2022 (Task Arithmetic)
- Yadav et al., 2023 (TIES-Merging)
- Jordan et al., 2022 (REPAIR)
- Yu et al., 2024 (DARE)
- He et al., 2016 (ResNet-18)
- Loshchilov & Hutter, 2019 (AdamW)
- torchvision, CIFAR-10, SVHN, FashionMNIST.

Output ONLY the BibTeX code for `paper.bib` without markdown code block backticks.
"""

response_bib = model.generate_content(bib_prompt)
with open("paper.bib", "w") as f:
    f.write(response_bib.text.strip())
print("paper.bib generated successfully.")
