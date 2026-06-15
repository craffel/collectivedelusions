import os
import subprocess

def main():
    os.makedirs("results", exist_ok=True)
    
    seeds = [43, 44]
    runs = [
        # Lambda = 0.0 runs
        {"optimizer": "adamw", "merging": "task_arithmetic", "lambda_val": 0.0},
        {"optimizer": "adamw", "merging": "isotropic", "lambda_val": 0.0},
        {"optimizer": "sam", "merging": "task_arithmetic", "lambda_val": 0.0},
        {"optimizer": "sam", "merging": "isotropic", "lambda_val": 0.0},
        # Lambda = 0.2 runs
        {"optimizer": "adamw", "merging": "task_arithmetic", "lambda_val": 0.2},
        {"optimizer": "adamw", "merging": "isotropic", "lambda_val": 0.2},
        {"optimizer": "sam", "merging": "task_arithmetic", "lambda_val": 0.2},
        {"optimizer": "sam", "merging": "isotropic", "lambda_val": 0.2},
        {"optimizer": "adamw", "merging": "norm_matching", "lambda_val": 0.2},
        {"optimizer": "sam", "merging": "norm_matching", "lambda_val": 0.2},
    ]
    
    print(f"Submitting {len(runs) * len(seeds)} jobs for multi-seed verification")
    
    submitted_count = 0
    for seed in seeds:
        for run in runs:
            opt = run["optimizer"]
            merg = run["merging"]
            lamb = run["lambda_val"]
            
            if lamb == 0.0:
                out_file = f"results/{opt}_{merg}_seed{seed}.json"
            else:
                out_file = f"results/{opt}_{merg}_lambda_02_seed{seed}.json"
                
            # Check if already exists
            if os.path.exists(out_file):
                print(f"Skipping {opt}_{merg} seed {seed} (exists)")
                continue
                
            print(f"Submitting job: {opt} | {merg} | Lambda {lamb} | Seed {seed}")
            
            slurm_script = f"""#!/bin/bash
#SBATCH --job-name=saim-seed-{seed}-{opt}-{merg}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH -o saim-seed-{seed}-{opt}-{merg}_%j.out
#SBATCH -e saim-seed-{seed}-{opt}-{merg}_%j.err

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
                print(f"Submitted seed {seed} successfully: {res.stdout.strip()}")
                submitted_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job:")
                print(e.stderr)
                
    print(f"Successfully submitted {submitted_count} jobs.")

if __name__ == "__main__":
    main()
