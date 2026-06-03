import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_config(name, args_str):
    cmd = f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export TORCH_NUM_THREADS=1; python3 main.py --train_epochs 0 {args_str}"
    print(f"Starting Sweep: {name} (Command: {cmd})")
    
    out_file = f"sweep_{name}.out"
    with open(out_file, "w") as f:
        res = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        
    print(f"Completed {name}! Output saved to {out_file}. Exit Code: {res.returncode}")
    return name, res.returncode

def main():
    # Verify checkpoints exist before starting
    checkpoints = ["mnist_expert.pt", "fashion_expert.pt", "cifar_expert.pt"]
    for cp in checkpoints:
        if not os.path.exists(cp):
            print(f"Error: Checkpoint {cp} not found yet. Please wait for the Slurm job to finish training.")
            return

    configs = [
        ("cal_n32", "--seed 42 --cal_size 32 --method WA"),
        ("cal_n128", "--seed 42 --cal_size 128 --method WA"),
        ("cal_n512", "--seed 42 --cal_size 512 --method WA"),
        ("ta_lam_10", "--seed 42 --cal_size 128 --method TA --lambda_val 0.1"),
        ("ta_lam_20", "--seed 42 --cal_size 128 --method TA --lambda_val 0.2"),
        ("ta_lam_30", "--seed 42 --cal_size 128 --method TA --lambda_val 0.3"),
        ("ta_lam_40", "--seed 42 --cal_size 128 --method TA --lambda_val 0.4"),
        ("seed_100", "--seed 100 --cal_size 128 --method WA"),
        ("seed_2026", "--seed 2026 --cal_size 128 --method WA")
    ]

    print(f"Running {len(configs)} sweeps in parallel...")
    
    # Use 3 workers to not overwhelm the 4-CPU system but still run fast
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_config, name, args_str): name for name, args_str in configs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                name, ret_code = future.result()
            except Exception as e:
                print(f"Sweep {name} raised an exception: {e}")

    print("\nAll parallel sweeps completed successfully!")

if __name__ == "__main__":
    main()
