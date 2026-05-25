import subprocess
import time
import os

def check_queue():
    res = subprocess.run("squeue", shell=True, capture_output=True, text=True)
    # Check if there are any jobs with the name "ttmm-exp"
    return "ttmm-exp" in res.stdout

def main():
    print("=== Monitoring adaptation jobs in Slurm queue... ===")
    
    # Wait for the jobs to start and finish
    while True:
        if not check_queue():
            print("All 'ttmm-exp' jobs have completed in the queue!")
            break
        print("Jobs are still running/pending in Slurm. Waiting 15 seconds...")
        time.sleep(15)
        
    print("\n=== All jobs finished! Checking for result files in results/ ===")
    results_dir = "results"
    methods = ["Static", "PROTO-TTMM", "IGGS-OW", "Rob-OW"]
    corruptions = ["clean", "corrupted"]
    
    missing = []
    for method in methods:
        for corr in corruptions:
            filename = f"result_{method}_{corr}_seed42.txt"
            if not os.path.exists(os.path.join(results_dir, filename)):
                missing.append(filename)
                
    if missing:
        print(f"WARNING: Some result files are missing: {missing}")
        print("Check files in logs/ directory to see if any job failed.")
        return
        
    print("All 8 results files exist. Generating plots and compiling paper...")
    
    # Generate plots
    print("\nRunning generate_plots.py...")
    res_plot = subprocess.run("python generate_plots.py", shell=True, capture_output=True, text=True)
    print(res_plot.stdout)
    if res_plot.stderr:
        print("Error during plot generation:", res_plot.stderr)
        
    # Compile PDF
    print("\nRunning parse_results_and_compile_pdf.py...")
    res_compile = subprocess.run("python parse_results_and_compile_pdf.py", shell=True, capture_output=True, text=True)
    print(res_compile.stdout)
    if res_compile.stderr:
        print("Error during compilation:", res_compile.stderr)
        
    print("\n=== Process completed! ===")
    if os.path.exists("submission.pdf"):
        print("SUCCESS: submission.pdf compiled successfully!")
    else:
        print("ERROR: submission.pdf not found.")

if __name__ == "__main__":
    main()
