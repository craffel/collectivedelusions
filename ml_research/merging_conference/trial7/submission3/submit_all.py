import subprocess

methods = ["Static", "PROTO-TTMM", "IGGS-OW", "Rob-OW"]
corruptions = ["clean", "corrupted"]

for method in methods:
    for corruption in corruptions:
        cmd = f"sbatch --export=ALL,METHOD={method},CORRUPTION={corruption} experiment.slurm"
        print(f"Executing: {cmd}")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(res.stdout.strip())
        if res.stderr:
            print(f"Error: {res.stderr.strip()}")
