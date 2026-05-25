import subprocess
import time

jobs = [
    # Baseline (AdamW)
    {"task": "A", "opt": "adamw", "checkpoint_a": "expert_a_adamw.pt", "checkpoint_b": "expert_b_adamw.pt", "name": "train-A-adamw"},
    {"task": "B", "opt": "adamw", "checkpoint_a": "expert_a_adamw.pt", "checkpoint_b": "expert_b_adamw.pt", "name": "train-B-adamw"},
    
    # Isotropic SAM
    {"task": "A", "opt": "sam", "checkpoint_a": "expert_a_sam.pt", "checkpoint_b": "expert_b_sam.pt", "name": "train-A-sam", "rho": 0.05},
    {"task": "B", "opt": "sam", "checkpoint_a": "expert_a_sam.pt", "checkpoint_b": "expert_b_sam.pt", "name": "train-B-sam", "rho": 0.05},
    
    # Task-Arithmetic-Aware Sharpness Regularization (TAA-SR - Ours)
    {"task": "A", "opt": "taa_sr", "checkpoint_a": "expert_a_taa_sr.pt", "checkpoint_b": "expert_b_taa_sr.pt", "name": "train-A-taasr", "rho": 0.05},
    {"task": "B", "opt": "taa_sr", "checkpoint_a": "expert_a_taa_sr.pt", "checkpoint_b": "expert_b_taa_sr.pt", "name": "train-B-taasr", "rho": 0.05}
]

job_ids = []
for job in jobs:
    cmd = [
        "sbatch",
        f"--job-name={job['name']}",
        "train.slurm",
        "--mode", "train",
        "--task", job["task"],
        "--opt", job["opt"],
        "--epochs", "5",
        "--batch_size", "128",
        "--lr", "0.005"
    ]
    
    if "rho" in job:
        cmd.extend(["--rho", str(job["rho"])])
    if job["task"] == "A":
        cmd.extend(["--checkpoint_a", job["checkpoint_a"]])
    else:
        cmd.extend(["--checkpoint_b", job["checkpoint_b"]])
        
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        output = result.stdout.strip()
        print(f"Success: {output}")
        # Parse job id from "Submitted batch job <job_id>"
        job_id = output.split()[-1]
        job_ids.append(job_id)
    else:
        print(f"Error: {result.stderr}")

print("\nAll jobs submitted successfully!")
print(f"Job IDs: {job_ids}")
