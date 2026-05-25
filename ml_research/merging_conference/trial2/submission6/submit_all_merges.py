import os
import subprocess

methods = ["standard", "sam", "sosr", "sata"]
merge_methods = ["dpm", "svdm", "palm", "lrom"]

os.makedirs("./slurm_scripts", exist_ok=True)
os.makedirs("./results", exist_ok=True)

for method in methods:
    ckpt1 = f"./models/cifar10_{method}.pt"
    ckpt2 = f"./models/svhn_{method}.pt"
    
    for merge_method in merge_methods:
        job_name = f"merge-{method}-{merge_method}"
        script_path = f"./slurm_scripts/{job_name}.slurm"
        
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH -o ./slurm_logs/%x_%j.out
#SBATCH -e ./slurm_logs/%x_%j.err

echo "Starting merge {method} {merge_method}"
/fsx/craffel/miniconda3/bin/python merge.py --ckpt1 {ckpt1} --ckpt2 {ckpt2} --merge_method {merge_method}
echo "Completed merge {method} {merge_method}"
"""
        with open(script_path, "w") as f:
            f.write(slurm_content)
            
        print(f"Submitting {job_name}...")
        res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        print(res.stdout.strip())
