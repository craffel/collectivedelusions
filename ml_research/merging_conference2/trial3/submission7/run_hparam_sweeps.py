import subprocess
import os

def launch_job(merge_method, ta_lam, rep_cal, head_align, cal_size, seed, lr, epochs):
    # Construct filename
    filename = f"{merge_method}_rep{rep_cal}_head{head_align}_N{cal_size}_seed{seed}_lr{lr}_epochs{epochs}"
    output_path = f"results/hparam_sweeps/{filename}.json"
    
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
        f"--sft_tta_lr {lr} "
        f"--epochs {epochs} "
        f"--output_file {output_path}"
    )
    
    # Slurm sbatch command
    sbatch_cmd = (
        f"sbatch --partition=hopper-prod --qos=low --nodes=1 --gpus-per-node=1 "
        f"--cpus-per-task=12 --time=0:25:00 "
        f"-o logs/hparam_sweeps/{filename}_%j.out -e logs/hparam_sweeps/{filename}_%j.err "
        f"--wrap=\"{cmd}\""
    )
    
    # Execute sbatch
    result = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error launching {filename}: {result.stderr}")
    else:
        print(f"Successfully launched: {filename}")

def main():
    # Ensure logs and results directories exist
    os.makedirs("logs/hparam_sweeps", exist_ok=True)
    os.makedirs("results/hparam_sweeps", exist_ok=True)
    
    # We will sweep LR and Epochs under REDA (N-TAAC + SFT and N-TAAC + TTA) with WA, N=128, seed=42
    lrs = [1e-4, 5e-4, 1e-3, 2e-3]
    epochs_list = [10, 15, 20]
    head_aligns = ["sft", "tta"]
    
    print("--- Queueing Hyperparameter Sweep Matrix ---")
    for lr in lrs:
        for epochs in epochs_list:
            for head_align in head_aligns:
                launch_job(
                    merge_method="wa",
                    ta_lam=0.0,
                    rep_cal="ntaac",
                    head_align=head_align,
                    cal_size=128,
                    seed=42,
                    lr=lr,
                    epochs=epochs
                )

if __name__ == "__main__":
    main()
