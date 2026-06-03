import os
import subprocess
import re

def submit_job(script_content, job_name, dependency=None):
    os.makedirs("slurm_scripts", exist_ok=True)
    script_path = f"slurm_scripts/{job_name}.slurm"
    with open(script_path, "w") as f:
        f.write(script_content)
        
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(script_path)
    
    print(f"Submitting {job_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
        
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        return match.group(1)
    return None

def main():
    # 1. Submit ViT-B/16 expert training
    vit_train_content = """#!/bin/bash
#SBATCH --job-name=train-vit-experts
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

echo "Starting ViT expert training..."
python train_experts.py --arch vit_b_16 --task cifar10 --epochs 5 --lr 1e-4 --batch_size 128
python train_experts.py --arch vit_b_16 --task svhn --epochs 5 --lr 1e-4 --batch_size 128
python train_experts.py --arch vit_b_16 --task fmnist --epochs 5 --lr 1e-4 --batch_size 128
echo "ViT expert training completed!"
"""
    vit_train_id = submit_job(vit_train_content, "train_vit_experts")
    print(f"ViT Train Job ID: {vit_train_id}")
    
    # Active ResNet-18 sweep job IDs
    resnet_job_ids = ["22161708", "22161709", "22161710", "22161711", "22161712", "22161713"]
    
    # 2. Submit ViT-B/16 Sweeps (dependent on vit_train_id)
    methods = ['arithmetic', 'ties', 'dare', 'orthomerge', 'saim', 'dor_saim']
    vit_job_ids = []
    if vit_train_id:
        for method in methods:
            content = f"""#!/bin/bash
#SBATCH --job-name=sweep-vit-{method}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

echo "Starting ViT-B/16 sweep for {method}..."
python run_method_sweep.py --arch vit_b_16 --method {method}
"""
            job_id = submit_job(content, f"sweep_vit_{method}", dependency=vit_train_id)
            if job_id:
                vit_job_ids.append(job_id)
                
    # 3. Submit Compilation Job (dependent on all active resnet18 and newly submitted vit_b_16 sweeps!)
    all_sweep_ids = resnet_job_ids + vit_job_ids
    if all_sweep_ids:
        compilation_dependency = ":".join(all_sweep_ids)
        content = f"""#!/bin/bash
#SBATCH --job-name=compile-all-results
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

echo "Compiling all sweep results..."
python compile_results.py
echo "All results compiled successfully!"
"""
        submit_job(content, "compile_all_results", dependency=compilation_dependency)

if __name__ == "__main__":
    main()
