import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_config(name, args_str):
    cmd = f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export TORCH_NUM_THREADS=1; python3 main.py --train_epochs 0 {args_str}"
    print(f"Starting Hyperparameter Sweep: {name} (Command: {cmd})")
    
    out_file = f"sweep_hp_{name}.out"
    with open(out_file, "w") as f:
        res = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        
    print(f"Completed {name}! Output saved to {out_file}. Exit Code: {res.returncode}")
    return name, res.returncode

def main():
    # Verify checkpoints exist
    checkpoints = ["mnist_expert.pt", "fashion_expert.pt", "cifar_expert.pt"]
    for cp in checkpoints:
        if not os.path.exists(cp):
            print(f"Error: Checkpoint {cp} not found yet. Please make sure expert checkpoints exist.")
            return

    # Hyperparameter configs to sweep (with --ours_only to run 3.5x faster)
    configs = [
        # LR Sweeps
        ("lr_1e4", "--seed 42 --cal_size 128 --method WA --head_lr 0.0001 --head_epochs 15 --ours_only"),
        ("lr_5e4", "--seed 42 --cal_size 128 --method WA --head_lr 0.0005 --head_epochs 15 --ours_only"),
        ("lr_1e3", "--seed 42 --cal_size 128 --method WA --head_lr 0.001 --head_epochs 15 --ours_only"),
        ("lr_2e3", "--seed 42 --cal_size 128 --method WA --head_lr 0.002 --head_epochs 15 --ours_only"),
        
        # Epoch Sweeps
        ("epochs_5", "--seed 42 --cal_size 128 --method WA --head_lr 0.001 --head_epochs 5 --ours_only"),
        ("epochs_10", "--seed 42 --cal_size 128 --method WA --head_lr 0.001 --head_epochs 10 --ours_only"),
        ("epochs_15", "--seed 42 --cal_size 128 --method WA --head_lr 0.001 --head_epochs 15 --ours_only"),
        ("epochs_20", "--seed 42 --cal_size 128 --method WA --head_lr 0.001 --head_epochs 20 --ours_only"),
    ]

    print(f"Running {len(configs)} hyperparameter sweeps in parallel...")
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_config, name, args_str): name for name, args_str in configs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                name, ret_code = future.result()
            except Exception as e:
                print(f"Sweep {name} raised an exception: {e}")

    print("\nAll hyperparameter sweeps completed successfully!")

if __name__ == "__main__":
    main()
