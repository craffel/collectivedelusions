import os
import subprocess

def main():
    os.makedirs("results", exist_ok=True)
    
    seeds = [42, 43, 44]
    optimizers = ["adamw", "sam"]
    
    runs = []
    for opt in optimizers:
        for seed in seeds:
            runs.append({
                "optimizer": opt,
                "merging": "scale_calibrated",
                "lambda_val": 0.2,
                "seed": seed
            })
            
    print(f"Submitting {len(runs)} configurations with lambda_val = 0.2 and merging = scale_calibrated")
    
    for run in runs:
        opt = run["optimizer"]
        merg = run["merging"]
        lamb = run["lambda_val"]
        seed = run["seed"]
        
        if seed == 42:
            out_file = f"results/{opt}_{merg}_lambda_02.json"
        else:
            out_file = f"results/{opt}_{merg}_lambda_02_seed{seed}.json"
            
        print(f"Submitting parallel job for Optimizer: {opt} | Merging: {merg} | Lambda: {lamb} | Seed: {seed}...")
        
        # Generate Slurm script content
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=saim-lambda-{opt}-{merg}-seed{seed}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH -o saim-lambda-{opt}-{merg}-seed{seed}_%j.out
#SBATCH -e saim-lambda-{opt}-{merg}-seed{seed}_%j.err

echo "=== Environment Setup ==="
source /fsx/craffel/miniconda3/bin/activate exp
export PYTHONPATH=.:$PYTHONPATH
export HUGGINGFACE_HUB_CACHE=/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial1/submission2/cache
export TORCH_HOME=/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial1/submission2/cache
export HF_HUB_OFFLINE=1

echo "=== Starting Unbuffered Python Run ==="
python -u train.py \\
    --model vit_tiny_patch16_224 \\
    --optimizer {opt} \\
    --merging {merg} \\
    --epochs 3 \\
    --batch_size 128 \\
    --lambda_val {lamb} \\
    --seed {seed} \\
    --output_file {out_file}
echo "=== Run Finished ==="
"""
        
        # Submit via stdin
        try:
            res = subprocess.run(
                ["sbatch", "--qos=low"],
                input=slurm_script,
                text=True,
                capture_output=True,
                check=True
            )
            print(f"Submitted successfully: {res.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job for {opt}_{merg}_seed{seed}:")
            print(e.stderr)

if __name__ == "__main__":
    main()
