import os
import subprocess

# List of configurations to run
configs = [
    # Baseline control
    {"mode": "orthomerge", "rho": 0.0, "lr": 1e-3, "name": "orthomerge_lr1e3"},
    
    # Decoupled SMM with different rhos
    {"mode": "smm_decoupled", "rho": 0.01, "lr": 1e-3, "name": "smm_dec_rho01_lr1e3"},
    {"mode": "smm_decoupled", "rho": 0.05, "lr": 1e-3, "name": "smm_dec_rho05_lr1e3"},
    {"mode": "smm_decoupled", "rho": 0.1, "lr": 1e-3, "name": "smm_dec_rho1_lr1e3"},
    {"mode": "smm_decoupled", "rho": 0.2, "lr": 1e-3, "name": "smm_dec_rho2_lr1e3"},
    {"mode": "smm_decoupled", "rho": 0.5, "lr": 1e-3, "name": "smm_dec_rho5_lr1e3"},
    
    # Joint SMM with different rhos
    {"mode": "smm_joint", "rho": 0.01, "lr": 1e-3, "name": "smm_joint_rho01_lr1e3"},
    {"mode": "smm_joint", "rho": 0.05, "lr": 1e-3, "name": "smm_joint_rho05_lr1e3"},
    {"mode": "smm_joint", "rho": 0.1, "lr": 1e-3, "name": "smm_joint_rho1_lr1e3"},
]

for cfg in configs:
    job_name = f"smm-{cfg['name']}"
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --time=0:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

python sweep_smm.py --mode {cfg['mode']} --rho {cfg['rho']} --lr {cfg['lr']} --save_path res_{cfg['name']}.json
"""
    
    script_filename = f"run_{cfg['name']}.slurm"
    with open(script_filename, "w") as f:
        f.write(slurm_script_content)
        
    print(f"Submitting job for {cfg['name']}...")
    # Run sbatch on the created file
    try:
        res = subprocess.run(["sbatch", script_filename], capture_output=True, text=True, check=True)
        print(res.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit {script_filename}: {e.stderr}")
