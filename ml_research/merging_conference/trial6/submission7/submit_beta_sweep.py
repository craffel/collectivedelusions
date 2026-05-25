import subprocess

configs = [
    {"name": "ttmm-beta-0.0", "args": ["--fisher_floor", "0.01", "--num_steps_per_batch", "1", "--beta_momentum", "0.0"]},
    {"name": "ttmm-beta-0.5", "args": ["--fisher_floor", "0.01", "--num_steps_per_batch", "1", "--beta_momentum", "0.5"]},
    {"name": "ttmm-beta-0.95", "args": ["--fisher_floor", "0.01", "--num_steps_per_batch", "1", "--beta_momentum", "0.95"]},
    {"name": "ttmm-beta-0.99", "args": ["--fisher_floor", "0.01", "--num_steps_per_batch", "1", "--beta_momentum", "0.99"]}
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
