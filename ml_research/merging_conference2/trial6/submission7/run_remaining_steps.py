import os
import time
import subprocess

print("Waiting for experiment_results.json to be generated...")
while not os.path.exists("experiment_results.json"):
    time.sleep(5)

print("Found experiment_results.json! Generating plots...")
subprocess.run(["python", "plot_results.py"], check=True)

print("Generating and compiling LaTeX paper...")
subprocess.run(["python", "compile_latex_on_gpu.py"], check=True)

print("All remaining steps completed successfully!")
