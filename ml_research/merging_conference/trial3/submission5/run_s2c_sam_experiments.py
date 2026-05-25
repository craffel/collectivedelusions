import subprocess
import time
import os

experiments = [
    # --- Clean Stream ---
    {"name": "clean-s2c-sam-r0.02-lr0.01", "args": ["--method", "s2c_sam", "--rho", "0.02", "--lr", "0.01"]},
    {"name": "clean-s2c-sam-r0.05-lr0.01", "args": ["--method", "s2c_sam", "--rho", "0.05", "--lr", "0.01"]},
    {"name": "clean-s2c-sam-r0.1-lr0.01", "args": ["--method", "s2c_sam", "--rho", "0.1", "--lr", "0.01"]},
    {"name": "clean-s2c-sam-r0.2-lr0.01", "args": ["--method", "s2c_sam", "--rho", "0.2", "--lr", "0.01"]},
    
    {"name": "clean-s2c-sam-r0.02-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.02", "--lr", "0.02"]},
    {"name": "clean-s2c-sam-r0.05-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.05", "--lr", "0.02"]},
    {"name": "clean-s2c-sam-r0.1-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.1", "--lr", "0.02"]},
    {"name": "clean-s2c-sam-r0.2-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.2", "--lr", "0.02"]},

    # --- Corrupted Stream: Gaussian Noise severity 0.15 ---
    {"name": "noise-s2c-sam-r0.02-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.02", "--lr", "0.02", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},
    {"name": "noise-s2c-sam-r0.05-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.05", "--lr", "0.02", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},
    {"name": "noise-s2c-sam-r0.1-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.1", "--lr", "0.02", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},
    {"name": "noise-s2c-sam-r0.2-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.2", "--lr", "0.02", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},

    # --- Corrupted Stream: Brightness shift 0.15 ---
    {"name": "bright-s2c-sam-r0.02-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.02", "--lr", "0.02", "--corruption", "brightness", "--corruption_severity", "0.15"]},
    {"name": "bright-s2c-sam-r0.05-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.05", "--lr", "0.02", "--corruption", "brightness", "--corruption_severity", "0.15"]},
    {"name": "bright-s2c-sam-r0.1-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.1", "--lr", "0.02", "--corruption", "brightness", "--corruption_severity", "0.15"]},
    {"name": "bright-s2c-sam-r0.2-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.2", "--lr", "0.02", "--corruption", "brightness", "--corruption_severity", "0.15"]},
]

launched_jobs = []

for exp in experiments:
    name = exp["name"]
    args = exp["args"]
    
    # Construct sbatch command
    cmd = ["sbatch", f"--job-name={name}", "run_tta.slurm"] + args
    print(f"Launching job {name} with command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        out_line = result.stdout.strip()
        print(f"Success: {out_line}")
        job_id = out_line.split()[-1]
        launched_jobs.append((name, job_id))
    else:
        print(f"Error launching {name}: {result.stderr.strip()}")
        
    time.sleep(0.5)

with open("experiments_tracker_sam.txt", "w") as f:
    for name, job_id in launched_jobs:
        f.write(f"{name},{job_id}\n")

print(f"\nSuccessfully launched {len(launched_jobs)} SAM sweep jobs. Tracked in experiments_tracker_sam.txt.")
