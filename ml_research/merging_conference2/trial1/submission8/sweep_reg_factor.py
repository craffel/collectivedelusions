import subprocess
import re
import numpy as np

def run_merge_command(reg_factor):
    cmd = [
        "python3", "merge.py",
        "--method", "orthomerge",
        "--spectral_method", "uniform",
        "--reg_factor", str(reg_factor),
        "--seed", "42"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
        
    output = res.stdout
    accuracies = {}
    matches = re.findall(r"Merged model accuracy on (\w+): ([\d\.]+)%", output)
    for dataset, acc in matches:
        accuracies[dataset] = float(acc)
        
    avg_match = re.search(r"Average Merged Accuracy: ([\d\.]+)%", output)
    if avg_match:
        accuracies["Average"] = float(avg_match.group(1))
        
    return accuracies

def main():
    print("==================================================")
    print("     SWEEPING REGULARIZATION FACTOR (REG_FACTOR)  ")
    print("==================================================\n")
    
    # Baseline: Task Arithmetic
    print("Running Task Arithmetic...")
    cmd = ["python3", "merge.py", "--method", "task_arithmetic", "--seed", "42"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    ta_avg = 0.0
    if res.returncode == 0:
        avg_match = re.search(r"Average Merged Accuracy: ([\d\.]+)%", res.stdout)
        if avg_match:
            ta_avg = float(avg_match.group(1))
            
    reg_factors = [0.0, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    results = {}
    
    for r in reg_factors:
        print(f"Running OrthoMerge with reg_factor={r}...")
        res = run_merge_command(r)
        if res:
            results[r] = res
            
    print("\n" + "="*80)
    print(f"{'Reg Factor':<15} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR10':<8} | {'SVHN':<8} | {'Average':<8}")
    print("="*80)
    print(f"{'Task Arithmetic':<15} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {ta_avg:<8.2f}")
    print("-"*80)
    for r in reg_factors:
        res = results.get(r)
        if res:
            print(f"{r:<15.1f} | {res.get('MNIST', 0.0):<8.2f} | {res.get('FashionMNIST', 0.0):<8.2f} | {res.get('CIFAR10', 0.0):<8.2f} | {res.get('SVHN', 0.0):<8.2f} | {res.get('Average', 0.0):<8.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
