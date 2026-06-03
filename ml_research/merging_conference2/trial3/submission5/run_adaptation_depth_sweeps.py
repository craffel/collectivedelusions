import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_config(name, args_str):
    cmd = f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export TORCH_NUM_THREADS=1; python3 main.py --train_epochs 0 {args_str}"
    print(f"Starting Adaptation Depth Sweep: {name} (Command: {cmd})")
    
    out_file = f"sweep_depth_{name}.out"
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

    # Adaptation depth configurations to sweep
    configs = [
        ("heads", "--seed 42 --cal_size 128 --method WA --adapt_depth heads"),
        ("layer4", "--seed 42 --cal_size 128 --method WA --adapt_depth layer4"),
        ("full", "--seed 42 --cal_size 128 --method WA --adapt_depth full"),
    ]

    print(f"Running {len(configs)} adaptation depth sweeps in parallel...")
    
    # We use 3 workers to run them concurrently on our 4-CPU system
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_config, name, args_str): name for name, args_str in configs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                name, ret_code = future.result()
            except Exception as e:
                print(f"Sweep {name} raised an exception: {e}")

    print("\nAll adaptation depth sweeps completed successfully!")

if __name__ == "__main__":
    main()
