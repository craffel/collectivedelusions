import os
import sys
import subprocess

def main():
    os.makedirs("results", exist_ok=True)
    
    runs = [
        # AdamW base optimizer
        {"optimizer": "adamw", "merging": "task_arithmetic"},
        {"optimizer": "adamw", "merging": "isotropic"},
        {"optimizer": "adamw", "merging": "spectral_dampening"},
        
        # Standard SAM
        {"optimizer": "sam", "merging": "task_arithmetic"},
        {"optimizer": "sam", "merging": "isotropic"},
        {"optimizer": "sam", "merging": "spectral_dampening"},
        
        # SABCD Literal (the exact formula from paper)
        {"optimizer": "sabcd_literal", "merging": "task_arithmetic"},
        {"optimizer": "sabcd_literal", "merging": "isotropic"},
        {"optimizer": "sabcd_literal", "merging": "spectral_dampening"},
        
        # SABCD Standard Adam (standard Adam on perturbed grad, restricted to Omega)
        {"optimizer": "sabcd_standard_adam", "merging": "task_arithmetic"},
        {"optimizer": "sabcd_standard_adam", "merging": "isotropic"},
        {"optimizer": "sabcd_standard_adam", "merging": "spectral_dampening"},
        
        # SABCD Adam GT (Adam on unperturbed grad, restricted to Omega)
        {"optimizer": "sabcd_adam_gt", "merging": "task_arithmetic"},
        {"optimizer": "sabcd_adam_gt", "merging": "isotropic"},
        {"optimizer": "sabcd_adam_gt", "merging": "spectral_dampening"},
    ]
    
    print(f"Total configurations in grid: {len(runs)}")
    
    submitted_count = 0
    for run in runs:
        opt = run["optimizer"]
        merg = run["merging"]
        out_file = f"results/{opt}_{merg}.json"
        
        # Check if already completed
        if os.path.exists(out_file):
            print(f"Skipping {opt}_{merg} - results already exist at {out_file}")
            continue
            
        print(f"Submitting parallel job for Optimizer: {opt} | Merging: {merg}...")
        
        # Generate Slurm script content
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=saim-{opt}-{merg}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH -o saim-{opt}-{merg}_%j.out
#SBATCH -e saim-{opt}-{merg}_%j.err

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
            submitted_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job for {opt}_{merg}:")
            print(e.stderr)
            
    print(f"\nSubmitted {submitted_count} parallel jobs with --qos=low.")

if __name__ == "__main__":
    main()
