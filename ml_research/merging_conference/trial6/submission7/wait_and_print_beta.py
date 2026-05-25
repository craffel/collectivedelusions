import glob
import os
import time

jobs = ["ttmm-beta-0.0", "ttmm-beta-0.5", "ttmm-beta-0.95", "ttmm-beta-0.99"]

def check_jobs():
    completed = 0
    for job in jobs:
        files = glob.glob(f"{job}_*.out")
        if not files:
            continue
        files.sort(key=os.path.getmtime)
        latest_file = files[-1]
        
        with open(latest_file, "r") as f:
            content = f.read()
        if "Completed!" in content:
            completed += 1
            
    return completed == len(jobs)

print("Waiting for beta sweep jobs to complete...")
start_time = time.time()
while time.time() - start_time < 600:
    if check_jobs():
        print("All beta sweep jobs completed successfully!")
        break
    time.sleep(15)
else:
    print("Timeout reached waiting for jobs to complete.")
