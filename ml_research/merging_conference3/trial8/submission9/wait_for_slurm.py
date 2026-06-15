import os
import subprocess
import time

job_id = os.environ.get("SLURM_JOB_ID")
if not job_id:
    print("SLURM_JOB_ID not found in environment. Exiting.")
    exit(0)

while True:
    try:
        output = subprocess.check_output(["squeue", "-h", "-j", job_id, "-O", "TimeLeft"]).decode().strip()
        print(f"Current time left: {output}")
        parts = output.split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            if minutes < 15:
                print(f"Time left is less than 15 minutes ({minutes}). Exiting loop.")
                break
        elif len(parts) == 3:
            # HH:MM:SS
            hours = int(parts[0])
            minutes = int(parts[1])
            total_minutes = hours * 60 + minutes
            if total_minutes < 15:
                print(f"Time left is less than 15 minutes ({total_minutes}). Exiting.")
                break
        else:
            print(f"Unknown format: {output}")
    except Exception as e:
        print(f"Error querying squeue: {e}")
    
    time.sleep(60)
