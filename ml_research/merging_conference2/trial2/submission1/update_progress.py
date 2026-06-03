import os
import json
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('models/gemini-2.5-pro')

# Load results summary
with open("results_summary.json", "r") as f:
    summary = json.load(f)

# Format summary table for markdown
table_rows = []
table_rows.append("| **Config (N, Layer)** | **AOS-CKA** | **AOS-MSE** | **AOS-Cosine** | **AOS-MMD** | **Oracle Peak** | **WA Baseline** |")
table_rows.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

for config, d in sorted(summary.items()):
    parts = config.split("_")
    size = parts[0].replace("size", "")
    layer = parts[1].replace("layer", "")
    config_name = f"N={size}, L={layer}"
    
    cka = f"{d['cka']['mean_acc']:.2f}% ± {d['cka']['std_acc']:.2f}%"
    mse = f"{d['mse']['mean_acc']:.2f}% ± {d['mse']['std_acc']:.2f}%"
    cosine = f"{d['cosine']['mean_acc']:.2f}% ± {d['cosine']['std_acc']:.2f}%"
    mmd = f"{d['mmd']['mean_acc']:.2f}% ± {d['mmd']['std_acc']:.2f}%"
    oracle = f"{d['oracle']['mean_acc']:.2f}% ± {d['oracle']['std_acc']:.2f}%"
    wa = f"{d['wa']['mean_acc']:.2f}%"
    
    row = f"| {config_name} | {cka} | {mse} | {cosine} | {mmd} | {oracle} | {wa} |"
    table_rows.append(row)

table_str = "\n".join(table_rows)

persona_text = """
You are "The Empiricist", an empirically driven researcher who is extremely good at running tons of experiments. Your research philosophy is that true progress in machine learning comes from exhaustive empirical validation and large-scale experimentation. Emphasize the breadth and depth of your empirical results.
"""

prompt = f"""
{persona_text}

Please generate a final, comprehensive, and complete `progress.md` file for our project.
The log must contain:
1. Introduction: adoption of the Empiricist persona and our objective.
2. Phase 1: Ideation & Selection (Detailing the 10 brainstormed ideas and how the seeded random number generator (seed 42, choice = 2) selected Idea #2: "Activation Overlap Search" (AOS)).
3. Phase 2: Experimentation (The complete empirical execution):
   - Training the 3 ResNet-18 task expert models on CIFAR-10 (69.45% accuracy), SVHN (81.50% accuracy), and FashionMNIST (88.34% accuracy).
   - Discovery of the critical BN statistics bug (subtraction/scaling of running statistics causes negative variance and NaN/collapse at lambdas > 0.3) and how we designed a highly robust averaging fix for BN buffers.
   - Launching the massive parallel sweeps (90 jobs) in parallel on Slurm using QoS low.
   - Presenting the complete, beautiful results table of our findings:
{table_str}
   - Detailed findings: CKA as a near-perfect proxy (reaching 58.43% ± 0.00% matching Oracle exactly at N=256), the extreme data efficiency of CKA down to N=16, and the complete failure of Weight Averaging (34.58%).
4. Phase 3: Manuscript Preparation:
   - Writing the standard-compliant, publication-ready `paper.tex` and `paper.bib` files.
   - Generating the visualization plots: `aos_alignment.png` (Oracle vs AOS curve alignment) and `aos_performance_vs_size.png` (Performance vs calibration size).

Ensure the tone is highly academic, direct, and strongly reflects our "Empiricist" identity of rigorous scaling, exhaustive sweeps, and robust evaluations across multiple random seeds.
Output ONLY the markdown content for `progress.md`.
"""

print("Calling Gemini to generate final progress.md...")
response = model.generate_content(prompt)
with open("progress.md", "w") as f:
    f.write(response.text.strip())
print("progress.md updated successfully.")
