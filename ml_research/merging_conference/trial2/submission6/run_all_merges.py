import os
import subprocess

methods = ["standard", "sam", "sosr", "sata"]
merge_methods = ["dpm", "svdm", "palm", "lrom"]

print("Creating automated merging runner...")

# Ensure output directory for results exists
os.makedirs("./results", exist_ok=True)

# We will run this script to execute all merges once the model checkpoints are available
for method in methods:
    ckpt1 = f"./models/cifar10_{method}.pt"
    ckpt2 = f"./models/svhn_{method}.pt"
    
    # Check if both checkpoints exist before running the merge
    if not (os.path.exists(ckpt1) and os.path.exists(ckpt2)):
        print(f"Checkpoints for {method} fine-tuning not found yet. Skipping...")
        continue
        
    for merge_method in merge_methods:
        print(f"\n==================================================")
        print(f"Merging {method} checkpoints using {merge_method}...")
        print(f"==================================================")
        
        # We call merge.py using subprocess
        cmd = [
            "/fsx/craffel/miniconda3/bin/python", "merge.py",
            "--ckpt1", ckpt1,
            "--ckpt2", ckpt2,
            "--merge_method", merge_method
        ]
        
        # Save output results directly in results folder under specialized names
        # e.g., results_sata_lrom.txt
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("Error:")
            print(res.stderr)
            
        # Move the standard output text file from merge.py to its distinct name
        source_res = f"./results/results_{merge_method}.txt"
        dest_res = f"./results/results_{method}_{merge_method}.txt"
        if os.path.exists(source_res):
            os.rename(source_res, dest_res)
            print(f"Renamed {source_res} to {dest_res}")
            
print("Automated merging runner setup complete.")
