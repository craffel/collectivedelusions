import os
import sys
import time
import subprocess
import json

def get_slurm_script(config_idx):
    script = f"""#!/bin/bash
#SBATCH --job-name=sata-op-{config_idx}
#SBATCH --partition=hopper-prod
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --time=1:00:00
#SBATCH -o sata-op-{config_idx}_%j.out
#SBATCH -e sata-op-{config_idx}_%j.err

echo "Starting configuration index {config_idx} on GPU cluster..."
date
python experiment.py --config_idx {config_idx}
echo "Completed configuration index {config_idx}!"
date
"""
    return script

def main():
    num_configs = 14
    print(f"============================================================")
    # 1. Check existing results
    print("Checking for existing result files...")
    existing_configs = []
    configs_to_run = []
    for i in range(num_configs):
        res_file = f"results_config_{i}.json"
        if os.path.exists(res_file):
            print(f"  [+] Found existing result file for configuration {i}: {res_file}")
            existing_configs.append(i)
        else:
            configs_to_run.append(i)
            
    print(f"Configurations to run: {configs_to_run}")
    
    # 2. Generate and submit Slurm scripts in parallel
    job_ids = []
    for i in configs_to_run:
        slurm_file = f"run_config_{i}.slurm"
        with open(slurm_file, "w") as f:
            f.write(get_slurm_script(i))
            
        print(f"Submitting job for configuration {i} (under --qos=low)...")
        # Submit the job and parse job ID
        result = subprocess.run(["sbatch", slurm_file], capture_output=True, text=True)
        if result.returncode == 0:
            # Output format is typically: "Submitted batch job 123456"
            output = result.stdout.strip()
            print(f"  {output}")
            job_id = output.split()[-1]
            job_ids.append(job_id)
        else:
            print(f"  ERROR submitting job for configuration {i}:")
            print(result.stderr)
            sys.exit(1)
            
    if job_ids:
        print(f"Submitted {len(job_ids)} jobs successfully!")
        print(f"Job IDs: {job_ids}")
        print(f"Waiting for results to be generated in parallel...")
    else:
        print("All configurations already have result files. Skipping job submission.")
    
    # 3. Poll for the presence of the JSON files that were submitted
    start_time = time.time()
    completed = [False] * num_configs
    # Pre-mark existing configs as completed
    for i in existing_configs:
        completed[i] = True
        
    while not all(completed):
        time.sleep(30)
        elapsed = time.time() - start_time
        print(f"Time elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s")
        
        # Check files
        for i in range(num_configs):
            if not completed[i]:
                res_file = f"results_config_{i}.json"
                if os.path.exists(res_file):
                    completed[i] = True
                    print(f"  [+] Config {i} finished! File found: {res_file}")
                    
        # Optional: Print squeue status of active jobs
        if job_ids:
            active_jobs = []
            squeue_res = subprocess.run(["squeue"], capture_output=True, text=True)
            for job_id in job_ids:
                if job_id in squeue_res.stdout:
                    active_jobs.append(job_id)
            if active_jobs:
                print(f"  Active Slurm jobs: {active_jobs}")
            
    print("\nAll configurations have completed training and evaluation!")
    
    # 4. Merge JSON files
    print("Merging results...")
    merged_results = {}
    for i in range(num_configs):
        res_file = f"results_config_{i}.json"
        with open(res_file, "r") as f:
            cfg_res = json.load(f)
        merged_results.update(cfg_res)
        
    with open("results.json", "w") as f:
        json.dump(merged_results, f, indent=4)
    print("Successfully wrote combined results to results.json!")
    
    # 5. Run only-plot mode
    print("Regenerating plots and summaries...")
    subprocess.run(["python", "experiment.py", "--only_plot"])
    
    # 6. Clean up temporary slurm files
    print("Cleaning up temporary Slurm scripts...")
    for i in range(num_configs):
        slurm_file = f"run_config_{i}.slurm"
        if os.path.exists(slurm_file):
            os.remove(slurm_file)
            
    print("Orchestration pipeline completed successfully!")

if __name__ == "__main__":
    main()
