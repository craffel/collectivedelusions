import os
import subprocess

def main():
    os.makedirs("results", exist_ok=True)
    
    runs = [
        {"optimizer": "adamw", "merging": "norm_matching", "lambda_val": 0.2},
        {"optimizer": "sam", "merging": "norm_matching", "lambda_val": 0.2},
    ]
    
    print(f"Submitting {len(runs)} configurations with lambda_val = 0.2 and merging = norm_matching")
    
    for run in runs:
        opt = run["optimizer"]
        merg = run["merging"]
        lamb = run["lambda_val"]
        out_file = f"results/{opt}_{merg}_lambda_02.json"
        
        print(f"Submitting parallel job for Optimizer: {opt} | Merging: {merg} | Lambda: {lamb}...")
        
        # Generate Slurm script content
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=saim-lambda-{opt}-{merg}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH -o saim-lambda-{opt}-{merg}_%j.out
#SBATCH -e saim-lambda-{opt}-{merg}_%j.err

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
            print(f"Error submitting job for {opt}_{merg}:")
            print(e.stderr)

if __name__ == "__main__":
    main()
