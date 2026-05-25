import subprocess
import time
import os

def main():
    print("=== STEP 1: Launching all 8 parallel Slurm jobs... ===")
    res_launch = subprocess.run("python submit_all.py", shell=True, capture_output=True, text=True)
    print(res_launch.stdout)
    if res_launch.stderr:
        print("Error during launch:", res_launch.stderr)
        
    print("\n=== STEP 2: Waiting for results and generating plots/compiling PDF... ===")
    print("This will wait until all 8 result files are written to results/ directory...")
    
    # Run generate_plots.py which waits and plots
    res_plot = subprocess.run("python generate_plots.py", shell=True, capture_output=True, text=True)
    print(res_plot.stdout)
    if res_plot.stderr:
        print("Error during plot generation:", res_plot.stderr)
        
    # Run parse_results_and_compile_pdf.py which waits, updates LaTeX and compiles
    res_compile = subprocess.run("python parse_results_and_compile_pdf.py", shell=True, capture_output=True, text=True)
    print(res_compile.stdout)
    if res_compile.stderr:
        print("Error during compilation:", res_compile.stderr)
        
    print("\n=== Orchestration Completed! ===")
    if os.path.exists("submission.pdf"):
        print("SUCCESS: submission.pdf is compiled and ready!")
    else:
        print("FAILED: submission.pdf was not created.")

if __name__ == "__main__":
    main()
