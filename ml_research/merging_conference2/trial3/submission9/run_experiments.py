import os
import subprocess
import csv
import json

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(result.stderr)
        return None
    return result.stdout

def parse_output(output):
    if not output:
        return None
    lines = output.strip().split("\n")
    results = {}
    avg_acc = 0.0
    for line in lines:
        if "Task: " in line and " | Accuracy: " in line:
            parts = line.split(" | Accuracy: ")
            task = parts[0].split("Task: ")[1].strip()
            acc = float(parts[1].split("%")[0].strip())
            results[task] = acc
        if "Average Accuracy:" in line:
            avg_acc = float(line.split("Average Accuracy:")[1].split("%")[0].strip())
    results["average"] = avg_acc
    return results

def main():
    methods = [
        {"name": "base", "args": "--method base"},
        {"name": "expert", "args": "--method expert"},
        {"name": "wa", "args": "--method wa"},
        # Task Arithmetic Sweeps
        {"name": "ta_0.1", "args": "--method ta --ta_lambda 0.1"},
        {"name": "ta_0.2", "args": "--method ta --ta_lambda 0.2"},
        {"name": "ta_0.3", "args": "--method ta --ta_lambda 0.3"},
        {"name": "ta_0.4", "args": "--method ta --ta_lambda 0.4"},
        {"name": "ta_0.5", "args": "--method ta --ta_lambda 0.5"},
        {"name": "ta_0.6", "args": "--method ta --ta_lambda 0.6"},
        # STDFS Sweeps
        {"name": "stdfs_0.01", "args": "--method stdfs --stdfs_low_freq 0.01"},
        {"name": "stdfs_0.05", "args": "--method stdfs --stdfs_low_freq 0.05"},
        {"name": "stdfs_0.1", "args": "--method stdfs --stdfs_low_freq 0.1"},
        {"name": "stdfs_0.2", "args": "--method stdfs --stdfs_low_freq 0.2"},
        {"name": "stdfs_0.3", "args": "--method stdfs --stdfs_low_freq 0.3"},
        {"name": "stdfs_0.5", "args": "--method stdfs --stdfs_low_freq 0.5"},
        # DARE Sweeps
        {"name": "dare_0.1", "args": "--method dare --drop_rate 0.1"},
        {"name": "dare_0.2", "args": "--method dare --drop_rate 0.2"},
        {"name": "dare_0.3", "args": "--method dare --drop_rate 0.3"},
        {"name": "dare_0.5", "args": "--method dare --drop_rate 0.5"},
        # TIES Sweeps
        {"name": "ties_0.1", "args": "--method ties --keep_ratio 0.1"},
        {"name": "ties_0.2", "args": "--method ties --keep_ratio 0.2"},
        {"name": "ties_0.3", "args": "--method ties --keep_ratio 0.3"},
        {"name": "ties_0.5", "args": "--method ties --keep_ratio 0.5"},
        # Spectral DARE Sweeps
        {"name": "s_dare_0.1", "args": "--method s_dare --drop_rate 0.1"},
        {"name": "s_dare_0.2", "args": "--method s_dare --drop_rate 0.2"},
        {"name": "s_dare_0.3", "args": "--method s_dare --drop_rate 0.3"},
        {"name": "s_dare_0.5", "args": "--method s_dare --drop_rate 0.5"},
        # Spectral TIES Sweeps
        {"name": "s_ties_0.1", "args": "--method s_ties --keep_ratio 0.1"},
        {"name": "s_ties_0.2", "args": "--method s_ties --keep_ratio 0.2"},
        {"name": "s_ties_0.3", "args": "--method s_ties --keep_ratio 0.3"},
        {"name": "s_ties_0.5", "args": "--method s_ties --keep_ratio 0.5"},
    ]
    
    head_settings = ["task_specific", "shared"]
    
    results_data = []
    
    for head in head_settings:
        print("="*60)
        print(f"RUNNING EXPERIMENTS FOR HEAD SETTING: {head.upper()}")
        print("="*60)
        for m in methods:
            # For base and expert, head setting doesn't affect the backbone merge logic
            if m["name"] in ["base", "expert"] and head == "shared":
                continue
                
            cmd = f"python merge.py {m['args']} --head_setting {head}"
            stdout = run_cmd(cmd)
            metrics = parse_output(stdout)
            if metrics:
                row = {
                    "head_setting": head,
                    "method": m["name"],
                    "mnist": metrics.get("mnist", 0.0),
                    "fashion": metrics.get("fashion", 0.0),
                    "cifar10": metrics.get("cifar10", 0.0),
                    "average": metrics.get("average", 0.0)
                }
                results_data.append(row)
                print(f"Method: {m['name']} | MNIST: {row['mnist']:.2f} | Fashion: {row['fashion']:.2f} | CIFAR-10: {row['cifar10']:.2f} | Avg: {row['average']:.2f}")
                
    # Save results to CSV
    csv_file = "checkpoints/results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["head_setting", "method", "mnist", "fashion", "cifar10", "average"])
        writer.writeheader()
        writer.writerows(results_data)
        
    print(f"\nSaved all results to {csv_file}")

if __name__ == "__main__":
    main()
