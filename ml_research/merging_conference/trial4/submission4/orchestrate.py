import time
import subprocess
import os
import json

def get_job_status(job_id):
    try:
        res = subprocess.run(['squeue', '-h', '-j', str(job_id)], capture_output=True, text=True)
        lines = res.stdout.strip().split('\n')
        if len(lines) == 0 or lines[0] == '':
            return None
        parts = lines[0].split()
        if len(parts) >= 5:
            return parts[4] # ST column (R, PD, etc.)
        return 'UNKNOWN'
    except Exception as e:
        print(f"Error checking status for job {job_id}: {e}")
        return None

def wait_for_job(job_id):
    print(f"Waiting for job {job_id} to complete...")
    while True:
        status = get_job_status(job_id)
        if status is None:
            print(f"Job {job_id} is no longer in queue (completed).")
            break
        print(f"Job {job_id} is currently in state: {status}. Sleeping 20 seconds...")
        time.sleep(20)

def main():
    train_job_id = 22159198
    
    # Wait for expert training to complete
    wait_for_job(train_job_id)
    
    # Check if expert checkpoints were saved successfully
    all_saved = True
    for idx in range(3):
        if not os.path.exists(f"checkpoints/expert_{idx}.pt") or not os.path.exists(f"checkpoints/fim_{idx}.pt"):
            all_saved = False
            
    if not all_saved:
        print("Error: Some expert checkpoints or FIMs were not saved successfully. Check train logs.")
        return
        
    print("All expert models and Fisher matrices are saved! Submitting TTA evaluation...")
    
    # Submit TTA evaluation job
    res = subprocess.run(['sbatch', 'run_tta.slurm'], capture_output=True, text=True)
    out = res.stdout.strip()
    print("Sbatch output:", out)
    
    if "Submitted batch job" not in out:
        print("Error: Failed to submit TTA evaluation job.")
        return
        
    tta_job_id = int(out.split()[-1])
    
    # Wait for TTA evaluation to complete
    wait_for_job(tta_job_id)
    
    print("\nTTA evaluation completed successfully!")
    
    # Read and print the evaluation results
    if os.path.exists('tta_results.json'):
        with open('tta_results.json', 'r') as f:
            results = json.load(f)
        print("\n=== FINAL TTA ACCURACY RESULTS ===")
        print(json.dumps(results, indent=4))
    else:
        print("Error: tta_results.json not found!")

if __name__ == "__main__":
    main()
