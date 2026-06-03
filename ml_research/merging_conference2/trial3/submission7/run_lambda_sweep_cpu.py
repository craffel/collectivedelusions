import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_config(merge_method, ta_lam, rep_cal, head_align, cal_size, seed, epochs=15):
    # Construct filename
    filename = f"{merge_method}_lam{ta_lam:.1f}_rep{rep_cal}_head{head_align}_N{cal_size}_seed{seed}"
    output_path = f"results/lambda_sweep/{filename}.json"
    
    # Check if results already exist to avoid redundant runs
    if os.path.exists(output_path):
        return f"Skipping {filename} (already exists)."
        
    cmd = [
        "python", "merge_and_align.py",
        "--merge_method", merge_method,
        "--ta_lam", f"{ta_lam:.2f}",
        "--rep_cal", rep_cal,
        "--head_align", head_align,
        "--cal_size", str(cal_size),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output_file", output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error on {filename}: {result.stderr}"
    else:
        return f"Completed: {filename}"

def main():
    os.makedirs("results/lambda_sweep", exist_ok=True)
    
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    seeds = [42, 100, 2026]
    
    # Configurations to evaluate: (rep_cal, head_align)
    configs = {
        "uncalibrated": ("none", "none"),
        "ntaac_only": ("ntaac", "none"),
        "tta_only": ("none", "tta"),
        "reda": ("ntaac", "tta")
    }
    
    tasks = []
    for seed in seeds:
        for ta_lam in lambdas:
            for name, (rep_cal, head_align) in configs.items():
                tasks.append({
                    "merge_method": "ta",
                    "ta_lam": ta_lam,
                    "rep_cal": rep_cal,
                    "head_align": head_align,
                    "cal_size": 128,
                    "seed": seed
                })
                
    print(f"Total configurations to run: {len(tasks)}")
    
    # Run in parallel using ProcessPoolExecutor
    completed_count = 0
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_config, **task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            completed_count += 1
            if "Completed" in res:
                print(f"[{completed_count}/{len(tasks)}] {res}")
            elif "Error" in res:
                print(f"[{completed_count}/{len(tasks)}] {res}")

if __name__ == "__main__":
    main()
