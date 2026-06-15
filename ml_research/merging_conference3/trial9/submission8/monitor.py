import os
import sys
import time
import subprocess
import json

def get_time_left():
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        print("Error: SLURM_JOB_ID environment variable not found.", flush=True)
        return None
    try:
        # Run squeue to get TimeLeft
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-O", "TimeLeft"],
            capture_output=True,
            text=True,
            check=True
        )
        time_str = result.stdout.strip()
        print(f"Current TimeLeft from squeue: {time_str}", flush=True)
        if not time_str:
            return None
        # Parse time_str. Format can be: "MM:SS", "HH:MM:SS", or "D-HH:MM:SS"
        parts = time_str.split("-")
        days = 0
        if len(parts) == 2:
            days = int(parts[0])
            time_part = parts[1]
        else:
            time_part = parts[0]
        
        time_parts = [int(p) for p in time_part.split(":")]
        if len(time_parts) == 3:
            hours, minutes, seconds = time_parts
        elif len(time_parts) == 2:
            hours = 0
            minutes, seconds = time_parts
        elif len(time_parts) == 1:
            hours = 0
            minutes = 0
            seconds = time_parts[0]
        else:
            return None
            
        total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except Exception as e:
        print(f"Exception while getting TimeLeft: {e}", flush=True)
        return None

def main():
    print("Starting background SLURM job time monitor...", flush=True)
    while True:
        seconds_left = get_time_left()
        if seconds_left is None:
            # If we cannot get the time, sleep and retry
            print("Could not retrieve SLURM time left, retrying in 30 seconds...", flush=True)
            time.sleep(30)
            continue
        
        print(f"Time left in job: {seconds_left} seconds ({seconds_left / 60:.2f} minutes)", flush=True)
        
        # If less than 15 minutes left (900 seconds), trigger final handoff
        if seconds_left <= 900:
            print("Less than 15 minutes left! Triggering final handoff.", flush=True)
            
            # Step 1: Re-compile LaTeX one final time to be absolutely sure
            try:
                print("Compiling final paper via tectonic...", flush=True)
                subprocess.run(
                    ["tectonic", "example_paper.tex"],
                    cwd="submission",
                    check=True
                )
                print("Successfully compiled final PDF.", flush=True)
                
                # Copy example_paper.pdf to submission.pdf and submission_draft.pdf
                subprocess.run(
                    ["cp", "submission/example_paper.pdf", "submission/submission.pdf"],
                    check=True
                )
                subprocess.run(
                    ["cp", "submission/example_paper.pdf", "submission/submission_draft.pdf"],
                    check=True
                )
                print("Synchronized PDF deliverables successfully.", flush=True)
            except Exception as e:
                print(f"Error during final compilation: {e}", flush=True)
            
            # Step 2: Update progress.json to "completed"
            try:
                with open("progress.json", "w") as f:
                    json.dump({"phase": "completed"}, f, indent=2)
                print("Successfully updated progress.json tocompleted.", flush=True)
            except Exception as e:
                print(f"Error updating progress.json: {e}", flush=True)
                
            break
        
        # Sleep for 60 seconds before checking again
        time.sleep(60)

if __name__ == "__main__":
    main()
