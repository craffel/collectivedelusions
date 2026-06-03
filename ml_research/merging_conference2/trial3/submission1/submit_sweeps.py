import argparse
import subprocess
import os

def submit_jobs(dependency_id=None):
    merge_modes = ["WA", "TA"]
    seeds = [42, 43, 44]
    cal_sizes = [4, 8, 16, 32, 64, 128, 256]
    ta_coeffs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    job_count = 0

    # Base sbatch command
    base_sbatch = [
        "sbatch",
        "--partition=hopper-prod",
        "--qos=low",
        "--nodes=1",
        "--gpus-per-node=1",
        "--cpus-per-task=10",
        "--time=0:15:00",
    ]

    if dependency_id:
        base_sbatch.append(f"--dependency=afterok:{dependency_id}")

    # 1. Submit Weight Averaging (WA) jobs
    for seed in seeds:
        for cal_size in cal_sizes:
            job_name = f"sweep_WA_0.3_N{cal_size}_S{seed}"
            cmd = base_sbatch + [
                f"--job-name={job_name}",
                "-o", f"logs/sweep_WA_0.3_N{cal_size}_S{seed}_%j.out",
                "-e", f"logs/sweep_WA_0.3_N{cal_size}_S{seed}_%j.err",
                "--wrap", f"python experiment.py --mode sweep --merge_mode WA --coeff 0.3 --cal_size {cal_size} --seed {seed}"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"Submitted WA job: {job_name} (ID: {res.stdout.strip().split()[-1]})")
                job_count += 1
            else:
                print(f"Failed to submit WA job: {job_name}")
                print(res.stderr)

    # 2. Submit Task Arithmetic (TA) jobs
    for seed in seeds:
        for coeff in ta_coeffs:
            for cal_size in cal_sizes:
                job_name = f"sweep_TA_{coeff}_N{cal_size}_S{seed}"
                cmd = base_sbatch + [
                    f"--job-name={job_name}",
                    "-o", f"logs/sweep_TA_{coeff}_N{cal_size}_S{seed}_%j.out",
                    "-e", f"logs/sweep_TA_{coeff}_N{cal_size}_S{seed}_%j.err",
                    "--wrap", f"python experiment.py --mode sweep --merge_mode TA --coeff {coeff} --cal_size {cal_size} --seed {seed}"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    print(f"Submitted TA job: {job_name} (ID: {res.stdout.strip().split()[-1]})")
                    job_count += 1
                else:
                    print(f"Failed to submit TA job: {job_name}")
                    print(res.stderr)

    print(f"\nSuccessfully submitted {job_count} sweep jobs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency", type=str, default=None, help="Slurm job ID dependency (afterok)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    submit_jobs(dependency_id=args.dependency)
