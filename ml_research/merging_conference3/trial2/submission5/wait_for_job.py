import os
import time
import subprocess

print("Waiting for experiment_metrics.json to be created...")
start_time = time.time()
while not os.path.exists("experiment_metrics.json"):
    # Check if the Slurm job is still running
    squeue_out = subprocess.check_output(["squeue"]).decode("utf-8")
    if "22255185" not in squeue_out:
        if os.path.exists("experiment_metrics.json"):
            break
        else:
            print("Error: Slurm job finished but experiment_metrics.json was not created!")
            exit(1)
            
    time.sleep(15)
    # Print status every 60 seconds
    elapsed = int(time.time() - start_time)
    if elapsed % 60 < 15:
        print(f"Still waiting... Elapsed time: {elapsed} seconds")
        
print("Success! experiment_metrics.json has been created.")
