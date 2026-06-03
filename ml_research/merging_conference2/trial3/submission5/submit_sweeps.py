import os
import subprocess

def submit_job(name, args_str, qos="low"):
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=hopper-prod
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --time=2:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

echo "--- Starting Sweep Job: {name} ---"
echo "Date: $(date)"
echo "Node: $SLURMD_NODENAME"

nvidia-smi

python3 -m pip install --user tqdm scikit-learn

echo "Running python3 main.py {args_str}"
python3 main.py {args_str}

echo "--- Sweep Job Completed ---"
"""
    filename = f"run_sweep_{name}.slurm"
    with open(filename, "w") as f:
        f.write(slurm_content)
        
    print(f"Submitting {name} slurm job...")
    res = subprocess.run(["sbatch", filename], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Error:", res.stderr)
        
    # Clean up local slurm file to keep workspace tidy
    os.remove(filename)

def main():
    # We will submit sweeps over:
    # 1. Calibration Sizes (N=32, N=512) for Weight Averaging
    submit_job("cal_n32", "--seed 42 --cal_size 32 --method WA", qos="low")
    submit_job("cal_n512", "--seed 42 --cal_size 512 --method WA", qos="low")
    
    # 2. Task Arithmetic (TA) with different lambdas
    for lam in [0.1, 0.2, 0.3, 0.4]:
        name = f"ta_lam_{int(lam*100)}"
        submit_job(name, f"--seed 42 --cal_size 128 --method TA --lambda_val {lam}", qos="low")
        
    # 3. Robustness over seeds (Seed 100, Seed 2026)
    submit_job("seed_100", "--seed 100 --cal_size 128 --method WA", qos="low")
    submit_job("seed_2026", "--seed 2026 --cal_size 128 --method WA", qos="low")

if __name__ == "__main__":
    main()
