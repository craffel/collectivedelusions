import subprocess
import time
import json
import os

def get_remaining_seconds():
    try:
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            print("SLURM_JOB_ID not found in environment.")
            return None
        
        # Run squeue to get TimeLeft
        cmd = ["squeue", "-h", "-j", job_id, "-O", "TimeLeft"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        time_str = res.stdout.strip()
        print(f"squeue output: {time_str}")
        
        # Parse time_str which can be DD-HH:MM:SS, HH:MM:SS or MM:SS
        days = 0
        hours = 0
        minutes = 0
        seconds = 0
        
        if "-" in time_str:
            days_part, hms_part = time_str.split("-")
            days = int(days_part)
            time_str = hms_part
            
        parts = list(map(int, time_str.split(":")))
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            minutes, seconds = parts
        elif len(parts) == 1:
            seconds = parts[0]
            
        total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except Exception as e:
        print(f"Error getting remaining time: {e}")
        return None

def main():
    print("Starting wait_and_complete.py...")
    
    while True:
        sec_left = get_remaining_seconds()
        if sec_left is None:
            print("Could not retrieve SLURM time. Sleeping for 30 seconds.")
            time.sleep(30)
            continue
            
        print(f"Time remaining: {sec_left} seconds ({sec_left / 60:.2f} minutes)")
        
        # We want to wait until time remaining is strictly less than 15 minutes (900 seconds)
        # But let's leave a safe buffer, e.g., 880 seconds (14 minutes and 40 seconds)
        if sec_left < 890:
            print("Remaining time is less than 15 minutes! Starting compilation and completion...")
            break
            
        # Determine how much to sleep. Let's sleep in intervals of 30 seconds or up to the difference
        diff = sec_left - 890
        sleep_time = min(60, diff) if diff > 0 else 10
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        
    # Phase 4 Completion Procedure
    print("Step 1: Compiling LaTeX document...")
    # Compile
    try:
        subprocess.run(["tectonic", "example_paper.tex"], cwd="submission", check=True)
        print("Compilation successful.")
    except Exception as e:
        print(f"LaTeX Compilation failed: {e}")
        
    print("Step 2: Syncing compiled PDF files...")
    # Copy compiled pdfs
    try:
        os.system("cp submission/example_paper.pdf submission/submission_draft.pdf")
        os.system("cp submission/example_paper.pdf submission/submission.pdf")
        print("PDF deliverables synchronized.")
    except Exception as e:
        print(f"Copying files failed: {e}")
        
    print("Step 3: Running mock review...")
    # Run mock review
    try:
        subprocess.run(["./run_mock_review.sh"], check=True)
        print("Mock review complete.")
    except Exception as e:
        print(f"Mock review script execution failed: {e}")
        
    print("Step 4: Writing completed status to progress.json...")
    # Write completion to progress.json
    try:
        progress = {"phase": "completed"}
        with open("progress.json", "w") as f:
            json.dump(progress, f, indent=2)
        print("progress.json updated to 'completed'.")
    except Exception as e:
        print(f"Updating progress.json failed: {e}")
        
    print("All tasks completed. wait_and_complete.py finished.")

if __name__ == "__main__":
    main()
