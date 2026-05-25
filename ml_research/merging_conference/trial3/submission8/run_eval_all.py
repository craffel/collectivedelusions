import subprocess
import json
import os

configs = [
    ("sgd", 0.0, 0.0),
    ("sam", 0.0, 0.0),
    ("spor", 0.05, 0.0),
    ("spor", 0.20, 0.0),
    ("fw_spor", 0.20, 0.1),
    ("fw_spor", 0.20, 0.5),
    ("fw_spor", 0.20, 1.0),
    ("fw_spor", 0.20, 2.0)
]

print("Running all evaluations...")
for regime, beta, gamma in configs:
    cmd = f"python experiment.py --mode eval --regime {regime} --beta {beta} --gamma {gamma}"
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Consolidate results into a beautiful Markdown table
print("\n" + "="*50)
print("FINAL CONSOLIDATED RESULTS")
print("="*50)

print("| Fine-Tuning Regime | Merging Method | Task A Acc (%) | Task B Acc (%) | Full CIFAR-10 Acc (%) | Avg. Res. Norm |")
print("|---|---|---|---|---|---|")

for regime, beta, gamma in configs:
    filename = f"results_{regime}_beta_{beta}_gamma_{gamma}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        regime_label = f"{regime.upper()}"
        if regime == "spor":
            regime_label += f" (beta={beta})"
        elif regime == "fw_spor":
            regime_label += f" (beta={beta}, gamma={gamma})"
            
        for merge_method in ["Task Arithmetic", "C-Ortho", "OM-All"]:
            metrics = data[merge_method]
            print(f"| {regime_label} | {merge_method} | {metrics['Acc A']:.2f} | {metrics['Acc B']:.2f} | {metrics['Full Acc']:.2f} | {metrics['Avg Res Norm']:.6f} |")
