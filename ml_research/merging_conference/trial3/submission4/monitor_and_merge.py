import os
import time
import subprocess
import json

def get_running_jobs():
    res = subprocess.run(["squeue", "-h", "-o", "%j"], capture_output=True, text=True)
    if res.returncode != 0:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip() if "train_" in line]

def main():
    configs = [
        ("sgd", 0.0),
        ("sam", 0.0),
        ("spor", 0.05),
        ("spor", 0.10),
        ("spor", 0.15),
        ("spor", 0.20),
        ("spor", 0.30),
        ("fg_spor_direct", 0.05),
        ("fg_spor_direct", 0.10),
        ("fg_spor_inverse", 0.05),
        ("fg_spor_inverse", 0.10),
        ("fg_spor_inverse", 0.15),
        ("fg_spor_inverse", 0.20),
        ("fg_spor_inverse", 0.30),
    ]

    print("Starting monitoring loop...")
    start_time = time.time()
    
    while True:
        # Check running slurm jobs
        running = get_running_jobs()
        print(f"Running/pending jobs: {len(running)}")
        
        # Check checkpoints
        checkpoints_ready = []
        for config, beta in configs:
            path_A = f"./checkpoints/expert_A_{config}_beta_{beta}.pth"
            path_B = f"./checkpoints/expert_B_{config}_beta_{beta}.pth"
            if os.path.exists(path_A) and os.path.exists(path_B):
                checkpoints_ready.append((config, beta))
                
        print(f"Checkpoints ready: {len(checkpoints_ready)}/{len(configs)}")
        
        # Run merge for ready checkpoints if they don't have summaries yet
        for config, beta in checkpoints_ready:
            summary_path = f"./checkpoints/summary_{config}_beta_{beta}.json"
            if not os.path.exists(summary_path):
                print(f"Running merge for {config} beta={beta}...")
                subprocess.run(["python", "merge.py", "--config", config, "--beta", str(beta)])
                
        if len(checkpoints_ready) == len(configs) and len(running) == 0:
            print("All checkpoints ready and no active jobs left!")
            break
            
        # Add safety timeout (e.g. 20 minutes)
        if time.time() - start_time > 1200:
            print("Timeout reached!")
            break
            
        time.sleep(15)

    # Consolidate results into a markdown table
    print("\nConsolidating results:")
    results = []
    for config, beta in configs:
        summary_path = f"./checkpoints/summary_{config}_beta_{beta}.json"
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                results.append(json.load(f))
                
    # Create Markdown table
    md = "# Experimental Results Summary\n\n"
    md += f"| Configuration | Beta | Exp A Acc | Exp B Acc | Procrustes Norm | TA (Task A / B / Full) | C-Ortho (Task A / B / Full) |\n"
    md += f"|---|---|---|---|---|---|---|\n"
    for r in results:
        config = r["config"]
        beta = r["beta"]
        exp_a = f"{r['expert_A_acc']:.2f}%"
        exp_b = f"{r['expert_B_acc']:.2f}%"
        norm = f"{r['procrustes_norm']:.6f}"
        ta = f"{r['ta']['task_A']:.2f}% / {r['ta']['task_B']:.2f}% / {r['ta']['full']:.2f}%"
        c_ortho = f"{r['c_ortho']['task_A']:.2f}% / {r['c_ortho']['task_B']:.2f}% / {r['c_ortho']['full']:.2f}%"
        md += f"| {config} | {beta} | {exp_a} | {exp_b} | {norm} | {ta} | {c_ortho} |\n"
        
    print(md)
    with open("results.md", "w") as f:
        f.write(md)
    print("Saved results to results.md")

if __name__ == "__main__":
    main()
