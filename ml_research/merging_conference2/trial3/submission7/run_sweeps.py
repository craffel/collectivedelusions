import subprocess
import os

def launch_job(merge_method, ta_lam, rep_cal, head_align, cal_size, seed, epochs=15):
    # Construct filename
    filename = f"{merge_method}_lam{ta_lam}_rep{rep_cal}_head{head_align}_N{cal_size}_seed{seed}"
    output_path = f"results/{filename}.json"
    
    # Check if results already exist to avoid redundant runs
    if os.path.exists(output_path):
        print(f"Skipping {filename} (already exists).")
        return
        
    # Command to run inside Slurm job
    cmd = (
        f"python merge_and_align.py "
        f"--merge_method {merge_method} "
        f"--ta_lam {ta_lam} "
        f"--rep_cal {rep_cal} "
        f"--head_align {head_align} "
        f"--cal_size {cal_size} "
        f"--seed {seed} "
        f"--epochs {epochs} "
        f"--output_file {output_path}"
    )
    
    # Slurm sbatch command
    sbatch_cmd = (
        f"sbatch --partition=hopper-prod --qos=low --nodes=1 --gpus-per-node=1 "
        f"--cpus-per-task=12 --time=0:25:00 "
        f"-o logs/{filename}_%j.out -e logs/{filename}_%j.err "
        f"--wrap=\"{cmd}\""
    )
    
    # Execute sbatch
    print(f"Launching: {filename}")
    result = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error launching job: {result.stderr}")
    else:
        print(f"Successfully launched: {result.stdout.strip()}")

def main():
    # Ensure logs and results directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    seeds = [42, 100, 2026]
    merge_methods = ["wa", "ta"]
    rep_cals = ["none", "ntaac", "lsc", "tsc"]
    head_aligns = ["none", "sft", "tta"]
    
    # Sweep 1: Main Matrix (2 merge methods x 4 calibration methods x 3 head alignment methods x 3 seeds)
    print("--- Queueing Main Sweep Matrix ---")
    for seed in seeds:
        for merge_method in merge_methods:
            ta_lam = 0.3 if merge_method == "ta" else 0.0
            for rep_cal in rep_cals:
                for head_align in head_aligns:
                    # Skip none + none which we can run locally or is redundant to run multiple times per seed
                    launch_job(
                        merge_method=merge_method,
                        ta_lam=ta_lam,
                        rep_cal=rep_cal,
                        head_align=head_align,
                        cal_size=128,
                        seed=seed
                    )
                    
    # Sweep 2: Sample Efficiency Sweep (seed 42, N in {16, 32, 64, 128, 256, 512} for key methods)
    print("\n--- Queueing Sample Efficiency Sweep ---")
    cal_sizes = [16, 32, 64, 256, 512] # 128 is already run in the main sweep
    key_configs = [
        # (rep_cal, head_align)
        ("none", "sft"),
        ("none", "tta"),
        ("ntaac", "none"),
        ("ntaac", "sft"),
        ("ntaac", "tta")
    ]
    
    for cal_size in cal_sizes:
        for rep_cal, head_align in key_configs:
            launch_job(
                merge_method="wa",
                ta_lam=0.0,
                rep_cal=rep_cal,
                head_align=head_align,
                cal_size=cal_size,
                seed=42
            )

if __name__ == "__main__":
    main()
