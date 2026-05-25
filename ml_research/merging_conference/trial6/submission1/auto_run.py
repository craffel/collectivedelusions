import subprocess
import time
import re
import os

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout, result.stderr, result.returncode

def monitor_job(job_id, name):
    print(f"Monitoring {name} job: {job_id}...")
    while True:
        stdout, _, _ = run_command("squeue")
        if str(job_id) not in stdout:
            print(f"{name} job {job_id} has finished.")
            break
        time.sleep(10)

def main():
    # 1. Wait for the training job 22159737 to finish
    train_job_id = 22159737
    monitor_job(train_job_id, "Expert Training")
    
    # Check that best checkpoints exist
    checkpoints = ["expert_cifar10.pth", "expert_svhn.pth", "expert_fmnist.pth"]
    for cp in checkpoints:
        cp_path = os.path.join("checkpoints", cp)
        if not os.path.exists(cp_path):
            print(f"Warning: Checkpoint {cp} not found at {cp_path}!")
        else:
            print(f"Found checkpoint: {cp} ({os.path.getsize(cp_path) / 1024 / 1024:.2f} MB)")
            
    # 2. Submit the run-experiments job
    print("Submitting experiments job...")
    stdout, stderr, code = run_command("sbatch run_experiments.slurm")
    if code != 0:
        print(f"Error submitting experiments job: {stderr}")
        return
        
    print(f"Submission output: {stdout.strip()}")
    # Extract job ID
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if not match:
        print("Failed to extract job ID from submission output.")
        return
        
    exp_job_id = int(match.group(1))
    
    # 3. Monitor the experiments job
    monitor_job(exp_job_id, "Experiments")
    
    # 4. Print the output log of the experiments job
    exp_log_pattern = f"run-experiments_{exp_job_id}.out"
    print(f"Searching for experiments log file: {exp_log_pattern}")
    if os.path.exists(exp_log_pattern):
        print("\n" + "="*80)
        print("                    EXPERIMENTS OUTPUT LOG                    ")
        print("="*80)
        with open(exp_log_pattern, "r") as f:
            print(f.read())
        print("="*80)
    else:
        # Fallback to look for run-experiments_* in current directory
        print("Log file not found directly, listing current run-experiments log files...")
        for f_name in os.listdir("."):
            if f_name.startswith("run-experiments_") and f_name.endswith(".out"):
                print(f"\n--- {f_name} ---")
                with open(f_name, "r") as f:
                    print(f.read())

if __name__ == "__main__":
    main()
