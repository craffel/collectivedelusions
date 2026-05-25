import subprocess
import time
import os

experiments = [
    # --- Blur Corruption Severity 0.5 ---
    {"name": "blur-static", "args": ["--method", "static", "--corruption", "blur", "--corruption_severity", "0.5"]},
    {"name": "blur-adamerging", "args": ["--method", "adamerging", "--corruption", "blur", "--corruption_severity", "0.5", "--lr", "0.02"]},
    {"name": "blur-standard", "args": ["--method", "standard_tta", "--corruption", "blur", "--corruption_severity", "0.5", "--lr", "0.02"]},
    {"name": "blur-sbf", "args": ["--method", "sbf_sat_symerge", "--corruption", "blur", "--corruption_severity", "0.5", "--lr", "0.02"]},
    {"name": "blur-s2c-g0.1", "args": ["--method", "s2c_merge", "--gamma", "0.1", "--corruption", "blur", "--corruption_severity", "0.5", "--lr", "0.02"]},
    {"name": "blur-s2c-g0.5", "args": ["--method", "s2c_merge", "--gamma", "0.5", "--corruption", "blur", "--corruption_severity", "0.5", "--lr", "0.02"]},
    {"name": "blur-s2c-sam-r0.02-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.02", "--lr", "0.02", "--corruption", "blur", "--corruption_severity", "0.5"]},
    {"name": "blur-s2c-sam-r0.05-lr0.02", "args": ["--method", "s2c_sam", "--rho", "0.05", "--lr", "0.02", "--corruption", "blur", "--corruption_severity", "0.5"]},
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

with open("experiments_tracker_blur.txt", "w") as f:
    for name, job_id in launched_jobs:
        f.write(f"{name},{job_id}\n")

print(f"\nSuccessfully launched {len(launched_jobs)} Blur TTA experiments. Tracked in experiments_tracker_blur.txt.")
