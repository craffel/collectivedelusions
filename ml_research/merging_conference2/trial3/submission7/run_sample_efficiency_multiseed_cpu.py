import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

def run_command(cmd_and_filename):
    cmd, filename = cmd_and_filename
    output_path = f"results/{filename}.json"
    
    # Skip if results already exist to avoid duplicate runs
    if os.path.exists(output_path):
        print(f"Skipping {filename} (already exists).")
        return
        
    print(f"Starting: {filename}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {filename}: {result.stderr}")
    else:
        print(f"Completed: {filename}")

def main():
    os.makedirs("results", exist_ok=True)
    
    seeds = [100, 2026] # 42 is already run
    cal_sizes = [16, 32, 64, 256, 512] # 128 is already run in main matrix
    key_configs = [
        # (rep_cal, head_align)
        ("none", "sft"),
        ("none", "tta"),
        ("ntaac", "none"),
        ("ntaac", "sft"),
        ("ntaac", "tta")
    ]
    
    tasks = []
    for seed in seeds:
        for cal_size in cal_sizes:
            for rep_cal, head_align in key_configs:
                filename = f"wa_lam0.0_rep{rep_cal}_head{head_align}_N{cal_size}_seed{seed}"
                cmd = (
                    f"python merge_and_align.py "
                    f"--merge_method wa "
                    f"--ta_lam 0.0 "
                    f"--rep_cal {rep_cal} "
                    f"--head_align {head_align} "
                    f"--cal_size {cal_size} "
                    f"--seed {seed} "
                    f"--epochs 15 "
                    f"--output_file results/{filename}.json"
                )
                tasks.append((cmd, filename))
                
    print(f"Total configurations to check/run: {len(tasks)}")
    
    # Run using ThreadPoolExecutor with 3 parallel workers
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(run_command, tasks)
        
    print("All configurations processed!")

if __name__ == "__main__":
    main()
