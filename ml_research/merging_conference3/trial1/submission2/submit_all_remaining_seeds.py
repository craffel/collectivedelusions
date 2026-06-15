import os
import subprocess

def main():
    os.makedirs("results", exist_ok=True)
    
    seeds = [43, 44]
    
    # 11 configurations
    configs = [
        {"optimizer": "adamw", "merging": "spectral_dampening"},
        {"optimizer": "sam", "merging": "spectral_dampening"},
        
        {"optimizer": "sabcd_literal", "merging": "task_arithmetic"},
        {"optimizer": "sabcd_literal", "merging": "isotropic"},
        {"optimizer": "sabcd_literal", "merging": "spectral_dampening"},
        
        {"optimizer": "sabcd_standard_adam", "merging": "task_arithmetic"},
        {"optimizer": "sabcd_standard_adam", "merging": "isotropic"},
        {"optimizer": "sabcd_standard_adam", "merging": "spectral_dampening"},
        
        {"optimizer": "sabcd_adam_gt", "merging": "task_arithmetic"},
        {"optimizer": "sabcd_adam_gt", "merging": "isotropic"},
        {"optimizer": "sabcd_adam_gt", "merging": "spectral_dampening"},
    ]
    
    runs = []
    for cfg in configs:
        for seed in seeds:
            runs.append({
                "optimizer": cfg["optimizer"],
                "merging": cfg["merging"],
                "lambda_val": 0.0,
                "seed": seed
            })
            
    print(f"Submitting {len(runs)} configurations for multi-seed verification of Table 1")
    
    for run in runs:
        opt = run["optimizer"]
        merg = run["merging"]
        lamb = run["lambda_val"]
        seed = run["seed"]
        
        out_file = f"results/{opt}_{merg}_seed{seed}.json"
            
        print(f"Submitting parallel job for Optimizer: {opt} | Merging: {merg} | Lambda: {lamb} | Seed: {seed}...")
        
        # Generate Slurm script content
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
            print(f"Submitted successfully: {res.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job for {opt}_{merg}_seed{seed}:")
            print(e.stderr)

if __name__ == "__main__":
    main()
