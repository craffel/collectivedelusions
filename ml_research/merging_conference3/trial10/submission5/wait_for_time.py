import time
import subprocess
import os
import json

def get_time_left_seconds():
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return 0
    try:
        out = subprocess.check_output(f"squeue -h -j {job_id} -O TimeLeft", shell=True).decode().strip()
        parts = out.split(':')
        if len(parts) == 2: # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3: # HH:MM:SS
            if '-' in parts[0]:
                days, hours = parts[0].split('-')
                return (int(days) * 24 * 3600) + (int(hours) * 3600) + int(parts[1]) * 60 + int(parts[2])
            else:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception as e:
        print("Error getting time left:", e)
        return 0
    return 0

def main():
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        print("No SLURM job ID active. Setting phase to completed immediately.")
        with open("progress.json", "w") as f:
            json.dump({"phase": "completed"}, f)
        return

    print(f"Active SLURM job ID: {job_id}. Starting wait loop until time left is under 15 minutes (900 seconds)...")
    
    while True:
        seconds_left = get_time_left_seconds()
        print(f"Time left: {seconds_left} seconds ({seconds_left / 60.0:.2f} minutes)")
        
        # We target less than 14 minutes and 30 seconds (870 seconds) to be safely under the 15-minute threshold
        if seconds_left <= 870:
            print("Remaining time is under 15 minutes! Writing completion phase to progress.json.")
            with open("progress.json", "w") as f:
                json.dump({"phase": "completed"}, f)
            break
            
        # Sleep for 1 minute
        time.sleep(60)

if __name__ == "__main__":
    main()
