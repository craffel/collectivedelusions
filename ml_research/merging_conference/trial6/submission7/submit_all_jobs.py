import subprocess

configs = [
    {"name": "ttmm-base", "args": ["--fisher_floor", "1e-6", "--num_steps_per_batch", "1"]},
    {"name": "ttmm-floor-0.01", "args": ["--fisher_floor", "0.01", "--num_steps_per_batch", "1"]},
    {"name": "ttmm-floor-0.05", "args": ["--fisher_floor", "0.05", "--num_steps_per_batch", "1"]},
    {"name": "ttmm-floor-0.1", "args": ["--fisher_floor", "0.1", "--num_steps_per_batch", "1"]},
    {"name": "ttmm-floor-0.05-steps-3", "args": ["--fisher_floor", "0.05", "--num_steps_per_batch", "3"]},
    {"name": "ttmm-floor-0.05-steps-5", "args": ["--fisher_floor", "0.05", "--num_steps_per_batch", "5"]}
]

for config in configs:
    cmd = [
        "sbatch",
        f"--job-name={config['name']}",
        f"--output={config['name']}_%j.out",
        f"--error={config['name']}_%j.err",
        "run_single_sweep.slurm"
    ] + config["args"]
    
    print(f"Submitting job for {config['name']}...")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout.strip())
