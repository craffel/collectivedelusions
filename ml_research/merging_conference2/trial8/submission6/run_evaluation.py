import subprocess
import os
import json

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
    return result

def main():
    print("Running evaluation on the 5 completed training configurations...")

    configs = [
        {"epochs": 5, "wd": 0.0},
        {"epochs": 5, "wd": 0.0001},
        {"epochs": 5, "wd": 0.01},
        {"epochs": 5, "wd": 0.1},
        {"epochs": 1, "wd": 0.01}
    ]

    bn_modes = ["uniform", "expert"]

    # Run evaluation
    for cfg in configs:
        epochs = cfg["epochs"]
        wd = cfg["wd"]
        for bn in bn_modes:
            cmd = f"python merge_and_evaluate.py --epochs {epochs} --weight_decay {wd} --seed 42 --bn_mode {bn}"
            run_cmd(cmd)

    # Parse and summarize results
    summary = []
    for cfg in configs:
        epochs = cfg["epochs"]
        wd = cfg["wd"]
        for bn in bn_modes:
            results_path = f"results/merge_results_epochs{epochs}_wd{wd}_seed{42}_bn{bn}.json"
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    data = json.load(f)
                
                sims = data["avg_similarities"]
                avg_sim = sum(sims.values()) / len(sims)
                res = data["results"]
                
                summary.append({
                    "epochs": epochs,
                    "wd": wd,
                    "bn_mode": bn,
                    "avg_similarity": avg_sim,
                    "wa": res["Weight Averaging (WA)"]["avg"],
                    "ta_05": res["Task Arithmetic (TA, lambda=0.5)"]["avg"],
                    "ta_07": res["Task Arithmetic (TA, lambda=0.7)"]["avg"],
                    "u_ipr": res["Update-level IPR (U-IPR)"]["avg"],
                    "hns": res["Holographic Norm Scaling (HNS)"]["avg"]
                })

    # Save summary as JSON
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Print summary table
    print("\n=== RESULTS SUMMARY TABLE ===")
    print("| Epochs | WD | BN Mode | Avg CosSim | WA | TA (0.5) | TA (0.7) | U-IPR | HNS |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in summary:
        print(f"| {row['epochs']} | {row['wd']} | {row['bn_mode']} | {row['avg_similarity']:.4f} | {row['wa']:.2f}% | {row['ta_05']:.2f}% | {row['ta_07']:.2f}% | {row['u_ipr']:.2f}% | {row['hns']:.2f}% |")

    # Generate report report_content
    report_content = "# Experimental Results & Methodology Stress-Test\n\n"
    report_content += "This document compiles the empirical results of our systematic sweep testing the foundational 'orthogonal task updates' assumption in model merging.\n\n"
    report_content += "## Merging Performance Sweep Table\n\n"
    report_content += "| Epochs | Weight Decay | BatchNorm Mode | Avg Update CosSim | WA (Avg Acc) | TA (0.5) | TA (0.7) | U-IPR | HNS |\n"
    report_content += "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for row in summary:
        report_content += f"| {row['epochs']} | {row['wd']} | {row['bn_mode']} | {row['avg_similarity']:.4f} | {row['wa']:.2f}% | {row['ta_05']:.2f}% | {row['ta_07']:.2f}% | {row['u_ipr']:.2f}% | {row['hns']:.2f}% |\n"
    
    with open("results_report.md", "w") as f:
        f.write(report_content)
    print("\nSaved summary report to results_report.md")

if __name__ == "__main__":
    main()
