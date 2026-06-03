import subprocess
import time
import os
import shutil

def wait_for_job(job_id):
    print(f"Waiting for Slurm job {job_id} to finish...")
    while True:
        try:
            output = subprocess.check_output("squeue", shell=True).decode()
            if str(job_id) not in output:
                print(f"Job {job_id} has finished!")
                break
        except Exception as e:
            print(f"Error checking queue: {e}")
        time.sleep(10)

def main():
    job_id = 22163702
    wait_for_job(job_id)
    
    # Run analyze_results.py locally just in case
    print("Running analysis locally to ensure plots are generated...")
    try:
        subprocess.check_call("python analyze_results.py", shell=True)
    except Exception as e:
        print(f"Error running analysis: {e}")
        
    # Copy paper.pdf to submission.pdf
    src = "template/paper.pdf"
    dst = "submission.pdf"
    
    if os.path.exists(src):
        print(f"Copying {src} to {dst}...")
        shutil.copy(src, dst)
        print("Copy complete!")
    else:
        print(f"Error: {src} not found! Regeneration paper...")
        try:
            subprocess.check_call("python generate_paper.py && cd template && tectonic paper.tex", shell=True)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print("Generated and copied successfully!")
            else:
                print("Failed to generate paper.pdf")
        except Exception as e:
            print(f"Error compiling paper: {e}")
            
    # Verify final PDF
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"SUCCESS: {dst} created successfully with size {os.path.getsize(dst)} bytes.")
    else:
        print(f"FAILURE: {dst} is missing or empty!")

if __name__ == "__main__":
    main()
