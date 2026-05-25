import subprocess

modes = ["sam", "so_lora_sam"]

for mode in modes:
    job_name = f"cifar10-{mode}"
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --time=1:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

echo "Resuming job for mode: {mode}"
python3 train_and_merge.py --mode {mode} --epochs 15 --batch_size 128
echo "Job finished!"
"""
    
    slurm_file = f"resume_{mode}.slurm"
    with open(slurm_file, "w") as f:
        f.write(slurm_content)
        
    print(f"Submitting {slurm_file}...")
    # Run sbatch
    res = subprocess.run(["sbatch", slurm_file], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Error:", res.stderr)
