import os
import subprocess

def create_slurm_script(task, config, beta):
    script_content = f"""#!/bin/bash
#SBATCH --job-name=train_{task}_{config}_beta_{beta}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

echo "Starting job train_{task}_{config}_beta_{beta}"
python train.py --task {task} --config {config} --beta {beta} --epochs 5
echo "Finished job train_{task}_{config}_beta_{beta}"
"""
    filename = f"slurm_scripts/train_{task}_{config}_beta_{beta}.slurm"
    with open(filename, "w") as f:
        f.write(script_content)
    return filename

def main():
    os.makedirs("slurm_scripts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    configs_to_sweep = [
        ("spor", 0.15),
        ("spor", 0.20),
        ("spor", 0.30),
        ("fg_spor_inverse", 0.15),
        ("fg_spor_inverse", 0.20),
        ("fg_spor_inverse", 0.30),
    ]

    job_ids = []
    for config, beta in configs_to_sweep:
        for task in ["A", "B"]:
            filename = create_slurm_script(task, config, beta)
            print(f"Created script: {filename}")
            
            # Submit job
            result = subprocess.run(["sbatch", filename], capture_output=True, text=True)
            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"Submitted successfully: {output}")
                # Extract job ID
                # Output format is usually "Submitted batch job 123456"
                parts = output.split()
                if parts:
                    job_ids.append(parts[-1])
            else:
                print(f"Failed to submit: {result.stderr.strip()}")

    print(f"\nSuccessfully submitted {len(job_ids)} jobs: {', '.join(job_ids)}")

if __name__ == "__main__":
    main()
