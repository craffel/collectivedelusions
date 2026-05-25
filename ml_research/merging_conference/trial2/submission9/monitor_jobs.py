import subprocess
import time
import sys

target_ids = ['22158365', '22158366', '22158367', '22158368', '22158369', '22158370']

print(f"Monitoring training jobs: {target_ids}")

while True:
    result = subprocess.run(["squeue"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error checking squeue: {result.stderr}")
        time.sleep(10)
        continue
        
    lines = result.stdout.strip().split("\n")
    active_jobs = {}
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 5:
            job_id = parts[0]
            status = parts[4]
            name = parts[2]
            if job_id in target_ids:
                active_jobs[job_id] = (name, status)
                
    if not active_jobs:
        print("All target training jobs have completed!")
        break
        
    print("\n--- Current Queue Status ---")
    for job_id, (name, status) in active_jobs.items():
        print(f"Job {job_id} ({name}): Status {status}")
        
    time.sleep(15)
