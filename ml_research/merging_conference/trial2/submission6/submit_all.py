import os
import subprocess

datasets_list = ["cifar10", "svhn"]
methods_list = ["standard", "sam", "sosr", "sata"]

os.makedirs("./slurm_scripts", exist_ok=True)
os.makedirs("./slurm_logs", exist_ok=True)

for dataset in datasets_list:
    for method in methods_list:
        job_name = f"train-{dataset}-{method}"
        script_path = f"./slurm_scripts/{job_name}.slurm"
        
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH -o ./slurm_logs/%x_%j.out
#SBATCH -e ./slurm_logs/%x_%j.err

echo "Starting job {job_name}"
/fsx/craffel/miniconda3/bin/python train_eval.py --dataset {dataset} --method {method} --epochs 3 --batch_size 128
echo "Job {job_name} completed"
"""
        with open(script_path, "w") as f:
            f.write(slurm_content)
            
        print(f"Submitting {job_name}...")
        # Submit to Slurm using the sbatch wrapper
        res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        print(res.stdout.strip())
        if res.stderr:
            print(f"Error submitting {job_name}: {res.stderr.strip()}")
