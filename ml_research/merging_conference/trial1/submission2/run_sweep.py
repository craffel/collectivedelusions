import os

rhos = [0.01, 0.02, 0.05]
lrs = [1e-3, 2e-3, 5e-3]

print("Launching hyperparameter sweep...")

for rho in rhos:
    for lr in lrs:
        job_name = f"usasla-sweep-rho_{rho}-lr_{lr}"
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:30:00
#SBATCH -o {job_name}_%j.out
#SBATCH -e {job_name}_%j.err

echo "=== Sweep Job Started: rho={rho}, lr={lr} ==="
date
python train_and_merge.py --rho {rho} --lr {lr} --output_file results_rho_{rho}_lr_{lr}.json
echo "=== Sweep Job Finished ==="
date
"""
        # Write temporary slurm script
        script_path = f"sweep_rho_{rho}_lr_{lr}.slurm"
        with open(script_path, "w") as f:
            f.write(slurm_content)
            
        # Submit the job
        cmd = f"sbatch {script_path}"
        print(f"Submitting: {cmd}")
        os.system(cmd)
        
        # Remove temporary slurm script to keep directory clean
        # (Wait, let's keep it or remove it, let's remove it to keep workspace clean)
        os.remove(script_path)
