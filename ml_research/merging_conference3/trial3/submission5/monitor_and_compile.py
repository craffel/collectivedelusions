import subprocess
import time
import os

def check_job_active(job_id):
    res = subprocess.run(f"squeue -h -j {job_id}", shell=True, capture_output=True, text=True)
    return len(res.stdout.strip()) > 0

def main():
    job_id = "22256541"
    print(f"Monitoring Slurm job {job_id}...")
    
    start_time = time.time()
    while check_job_active(job_id):
        elapsed = time.time() - start_time
        print(f"Job {job_id} is still running... (elapsed: {elapsed:.0f}s)")
        time.sleep(15)
        
    print(f"Slurm job {job_id} has completed successfully!")
    
    # 1. Run update_latex_results.py
    print("Running update_latex_results.py...")
    subprocess.run("python update_latex_results.py", shell=True, check=True)
    
    # 2. Compile LaTeX paper with tectonic
    print("Compiling LaTeX paper with tectonic...")
    subprocess.run("conda run -n latex tectonic -o submission/ submission/example_paper.tex", shell=True, check=True)
    
    # 3. Copy compiled PDF to submission/submission.pdf and submission/submission_draft.pdf
    print("Copying compiled PDF to submission targets...")
    subprocess.run("cp submission/example_paper.pdf submission/submission.pdf", shell=True, check=True)
    subprocess.run("cp submission/example_paper.pdf submission/submission_draft.pdf", shell=True, check=True)
    
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
