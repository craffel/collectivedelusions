import subprocess
import re
import numpy as np

def run_merge_command(method, spectral_method="uniform", gamma=1.0, reg_factor=1.0):
    cmd = [
        "python3", "merge.py",
        "--method", method,
        "--spectral_method", spectral_method,
        "--gamma", str(gamma),
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
    print("     EVALUATING SPECTRAL-AWARE ORTHOMERGE         ")
    print("==================================================\n")
    
    # 1. Baseline: Task Arithmetic
    print("Running Task Arithmetic baseline...")
    ta_results = run_merge_command("task_arithmetic")
    
    reg_values = [10.0, 50.0, 100.0]
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("\n" + "="*95)
    print(f"{'Method / Configuration':<45} | {'Reg':<6} | {'Gamma':<6} | {'MNIST':<6} | {'Fashion':<6} | {'CIFAR10':<6} | {'SVHN':<6} | {'Average':<6}")
    print("="*95)
    
    def print_row(name, reg, g, res):
        if res:
            print(f"{name:<45} | {reg:<6} | {g:<6} | {res.get('MNIST', 0.0):<6.2f} | {res.get('FashionMNIST', 0.0):<6.2f} | {res.get('CIFAR10', 0.0):<6.2f} | {res.get('SVHN', 0.0):<6.2f} | {res.get('Average', 0.0):<6.2f}")
            
    print_row("Task Arithmetic (Euclidean Baseline)", "-", "-", ta_results)
    print_row("OrthoMerge (Standard Uniform, reg=10.0)", "10.0", "1.0", run_merge_command("orthomerge", "uniform", 1.0, 10.0))
    print_row("OrthoMerge (Standard Uniform, reg=50.0)", "50.0", "1.0", run_merge_command("orthomerge", "uniform", 1.0, 50.0))
    print_row("OrthoMerge (Standard Uniform, reg=100.0)", "100.0", "1.0", run_merge_command("orthomerge", "uniform", 1.0, 100.0))
    print("-"*95)
    
    # Run Sweeps
    for reg in reg_values:
        for g in gammas:
            # Entropy-Weighted
            res = run_merge_command("orthomerge", "entropy", g, reg)
            print_row(f"SEW-OrthoMerge (Entropy-Weighted)", str(reg), str(g), res)
        print("-"*95)
        
    for reg in reg_values:
        for g in gammas:
            # Dominance-Weighted
            res = run_merge_command("orthomerge", "dominance", g, reg)
            print_row(f"SEW-OrthoMerge (Dominance-Weighted)", str(reg), str(g), res)
        print("-"*95)
        
    print("="*95)

if __name__ == "__main__":
    main()
