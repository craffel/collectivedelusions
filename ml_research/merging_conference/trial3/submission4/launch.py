import os
import subprocess

def launch_job(task, config, beta=0.05, epochs=5):
    job_name = f"train_{task}_{config}_beta_{beta}"
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

echo "Starting job {job_name}"
python train.py --task {task} --config {config} --beta {beta} --epochs {epochs}
echo "Finished job {job_name}"
"""
    os.makedirs("./slurm_scripts", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    script_path = f"./slurm_scripts/{job_name}.slurm"
    with open(script_path, "w") as f:
        f.write(slurm_script)
        
    print(f"Submitting job: {job_name}")
    # Run sbatch
    res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    print(res.stdout.strip())
    if res.stderr:
        print("Error:", res.stderr.strip())

def main():
    # We will launch 10 jobs (Task A and Task B for each configuration)
    configs = [
        ("sgd", 0.0),
        ("sam", 0.0),
        ("spor", 0.05),
        ("spor", 0.10),
        ("fg_spor_direct", 0.05),
        ("fg_spor_direct", 0.10),
        ("fg_spor_inverse", 0.05),
        ("fg_spor_inverse", 0.10),
    ]
    
    for config, beta in configs:
        for task in ["A", "B"]:
            launch_job(task, config, beta, epochs=5)

if __name__ == "__main__":
    main()
