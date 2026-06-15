import os
import subprocess

def main():
    os.makedirs("results", exist_ok=True)
    
    runs = [
        {"optimizer": "adamw", "merging": "task_arithmetic", "lambda_val": 0.0, "out_file": "results/vit_base_adamw_task_arithmetic_lambda_00.json"},
        {"optimizer": "sam", "merging": "task_arithmetic", "lambda_val": 0.0, "out_file": "results/vit_base_sam_task_arithmetic_lambda_00.json"},
        {"optimizer": "sam", "merging": "isotropic", "lambda_val": 0.2, "out_file": "results/vit_base_sam_isotropic_lambda_02.json"},
    ]
    
    print(f"Submitting {len(runs)} ViT-Base jobs to Slurm with --qos=low and /fsx/craffel/.cache for parallel scaling validation...")
    
    for run in runs:
        opt = run["optimizer"]
        merg = run["merging"]
        lamb = run["lambda_val"]
        out_file = run["out_file"]
        
        # Check if already exists (delete first to rerun)
        if os.path.exists(out_file):
            print(f"Deleting existing {out_file} to ensure fresh rerun.")
            os.remove(out_file)
            
        print(f"Submitting: {opt} | {merg} | Lambda {lamb} | {out_file}")
        
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=vit-base-{opt}-{merg}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH -o vit_base_{opt}_{merg}_lambda_{str(lamb).replace('.', '')}_%j.out
#SBATCH -e vit_base_{opt}_{merg}_lambda_{str(lamb).replace('.', '')}_%j.err

echo "=== Environment Setup ==="
source /fsx/craffel/miniconda3/bin/activate exp
export PYTHONPATH=.:$PYTHONPATH
export HUGGINGFACE_HUB_CACHE=/fsx/craffel/.cache
export TORCH_HOME=/fsx/craffel/.cache
export HF_HUB_OFFLINE=1

echo "=== Starting Unbuffered Python Run ==="
python -u train.py \\
    --model vit_base_patch16_224 \\
    --optimizer {opt} \\
    --merging {merg} \\
    --epochs 3 \\
    --batch_size 128 \\
    --lambda_val {lamb} \\
    --seed 42 \\
    --output_file {out_file}
echo "=== Run Finished ==="
"""
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
            print(f"Error submitting job:")
            print(e.stderr)

if __name__ == "__main__":
    main()
