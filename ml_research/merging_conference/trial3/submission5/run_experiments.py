import subprocess
import time
import os

experiments = [
    # --- Clean Stream ---
    {"name": "clean-static", "args": ["--method", "static"]},
    {"name": "clean-adamerging", "args": ["--method", "adamerging"]},
    {"name": "clean-standard", "args": ["--method", "standard_tta"]},
    {"name": "clean-sbf", "args": ["--method", "sbf_sat_symerge"]},
    {"name": "clean-s2c-g0.0", "args": ["--method", "s2c_merge", "--gamma", "0.0"]},
    {"name": "clean-s2c-g0.05", "args": ["--method", "s2c_merge", "--gamma", "0.05"]},
    {"name": "clean-s2c-g0.1", "args": ["--method", "s2c_merge", "--gamma", "0.1"]},
    {"name": "clean-s2c-g0.2", "args": ["--method", "s2c_merge", "--gamma", "0.2"]},
    {"name": "clean-s2c-g0.5", "args": ["--method", "s2c_merge", "--gamma", "0.5"]},
    {"name": "clean-s2c-g1.0", "args": ["--method", "s2c_merge", "--gamma", "1.0"]},

    # --- Corrupted Stream: Gaussian Noise severity 0.15 ---
    {"name": "noise-static", "args": ["--method", "static", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},
    {"name": "noise-adamerging", "args": ["--method", "adamerging", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},
    {"name": "noise-standard", "args": ["--method", "standard_tta", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},
    {"name": "noise-sbf", "args": ["--method", "sbf_sat_symerge", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},
    {"name": "noise-s2c-g0.1", "args": ["--method", "s2c_merge", "--gamma", "0.1", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},
    {"name": "noise-s2c-g0.5", "args": ["--method", "s2c_merge", "--gamma", "0.5", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},
    {"name": "noise-s2c-g1.0", "args": ["--method", "s2c_merge", "--gamma", "1.0", "--corruption", "gaussian_noise", "--corruption_severity", "0.15"]},

    # --- Corrupted Stream: Brightness shift 0.15 ---
    {"name": "bright-static", "args": ["--method", "static", "--corruption", "brightness", "--corruption_severity", "0.15"]},
    {"name": "bright-adamerging", "args": ["--method", "adamerging", "--corruption", "brightness", "--corruption_severity", "0.15"]},
    {"name": "bright-s2c-g0.1", "args": ["--method", "s2c_merge", "--gamma", "0.1", "--corruption", "brightness", "--corruption_severity", "0.15"]},
    {"name": "bright-s2c-g0.5", "args": ["--method", "s2c_merge", "--gamma", "0.5", "--corruption", "brightness", "--corruption_severity", "0.15"]},
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
        # Expected output format: "Submitted batch job 123456"
        out_line = result.stdout.strip()
        print(f"Success: {out_line}")
        job_id = out_line.split()[-1]
        launched_jobs.append((name, job_id))
    else:
        print(f"Error launching {name}: {result.stderr.strip()}")
        
    # Brief sleep to avoid hitting submission limits/overwhelming controller
    time.sleep(0.5)

# Write the mapping of names to job IDs to a file for tracking
with open("experiments_tracker.txt", "w") as f:
    for name, job_id in launched_jobs:
        f.write(f"{name},{job_id}\n")

print(f"\nSuccessfully launched {len(launched_jobs)} jobs. Tracked in experiments_tracker.txt.")
