import os
import subprocess
import time

sizes = [8, 16, 32, 64, 128, 256]
layers = ["layer3", "layer4", "avgpool"]
seeds = [42, 43, 44, 45, 46]

print("Starting submission of AOS experimental sweep (90 jobs) onto QoS low...")

for size in sizes:
    for layer in layers:
        for seed in seeds:
            # Check if output results file already exists to prevent duplicate runs
            res_file = f"results_size{size}_layer{layer}_seed{seed}.json"
            if os.path.exists(res_file):
                print(f"Results file {res_file} already exists, skipping.")
                continue
                
            slurm_content = f"""#!/bin/bash
#SBATCH --job-name=aos_sz{size}_ly{layer}_sd{seed}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:15:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

echo "Running merge_and_evaluate.py with size={size}, layer={layer}, seed={seed}"
python merge_and_evaluate.py {size} {layer} {seed}
echo "Job completed!"
"""
            # Write to a temp slurm script
            temp_filename = f"temp_aos_{size}_{layer}_{seed}.slurm"
            with open(temp_filename, "w") as f:
                f.write(slurm_content)
                
            # Submit using sbatch
            res = subprocess.run(["sbatch", temp_filename], capture_output=True, text=True)
            if res.returncode == 0:
                print(f"Submitted size={size}, layer={layer}, seed={seed} -> {res.stdout.strip()}")
            else:
                print(f"Failed to submit size={size}, layer={layer}, seed={seed}: {res.stderr}")
                
            # Clean up the temp slurm file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            # Be polite to the scheduler
            time.sleep(0.1)

print("All sweep jobs processed!")
