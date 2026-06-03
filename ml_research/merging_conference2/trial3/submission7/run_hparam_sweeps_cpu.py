import subprocess
import os
import concurrent.futures

def run_config(head_align, lr, epochs):
    filename = f"wa_repntaac_head{head_align}_N128_seed42_lr{lr}_epochs{epochs}"
    output_path = f"results/hparam_sweeps/{filename}.json"
    
    if os.path.exists(output_path):
        print(f"Skipping {filename} (already exists).")
        return
        
    cmd = [
        "python", "merge_and_align.py",
        "--merge_method", "wa",
        "--rep_cal", "ntaac",
        "--head_align", head_align,
        "--cal_size", "128",
        "--seed", "42",
        "--sft_tta_lr", str(lr),
        "--epochs", str(epochs),
        "--output_file", output_path
    ]
    
    print(f"Running locally on CPU: {filename}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {filename}: {result.stderr}")
    else:
        print(f"Completed: {filename}")

def main():
    os.makedirs("results/hparam_sweeps", exist_ok=True)
    
    lrs = [1e-4, 5e-4, 1e-3, 2e-3]
    epochs_list = [10, 15, 20]
    head_aligns = ["sft", "tta"]
    
    configs = []
    for lr in lrs:
        for epochs in epochs_list:
            for head_align in head_aligns:
                configs.append((head_align, lr, epochs))
                
    # Run with 2 workers to avoid CPU thrashing on a 4-CPU node
    print(f"Starting {len(configs)} configurations on 2 local CPU workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_config, head_align, lr, epochs) for head_align, lr, epochs in configs]
        concurrent.futures.wait(futures)
        
    print("All local CPU hyperparameter sweep runs completed!")

if __name__ == "__main__":
    main()
