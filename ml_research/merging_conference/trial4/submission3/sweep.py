import os
import subprocess

lr_lambdas = [0.1, 0.2, 0.5]
lr_heads = [1e-4, 5e-4, 1e-3]
gamma_regs = [1.0, 10.0, 100.0]

count = 0
for lr_l in lr_lambdas:
    for lr_h in lr_heads:
        for gr in gamma_regs:
            suffix = f"l{lr_l}_h{lr_h}_g{gr}".replace(".", "_")
            job_name = f"sweep_{suffix}"
            
            # Construct a dynamic slurm script content
            slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH -o {job_name}_%j.out
#SBATCH -e {job_name}_%j.err

python evaluate_tta.py --lr_lambda {lr_l} --lr_head {lr_h} --gamma_reg {gr} --num_mc_passes 5 --save_suffix {suffix}
"""
            # Write a temporary slurm script
            script_filename = f"run_{suffix}.slurm"
            with open(script_filename, "w") as f:
                f.write(slurm_content)
                
            # Submit using sbatch
            print(f"Submitting job for lr_lambda={lr_l}, lr_head={lr_h}, gamma_reg={gr}")
            subprocess.run(["sbatch", script_filename])
            
            # Clean up the script file immediately after submission
            os.remove(script_filename)
            count += 1

print(f"Submitted {count} low-priority sweep jobs!")
