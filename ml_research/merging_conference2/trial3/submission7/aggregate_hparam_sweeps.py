import os
import json
import glob
import numpy as np

def main():
    results = []
    # Find all json files in results/hparam_sweeps
    for path in glob.glob("results/hparam_sweeps/*.json"):
        if "test_cpu" in path:
            continue
        filename = os.path.basename(path).replace(".json", "")
        # Format: wa_repntaac_head{head_align}_N128_seed42_lr{lr}_epochs{epochs}
        parts = filename.split("_")
        if len(parts) < 7:
            continue
        
        head_align = parts[2].replace("head", "")
        lr = float(parts[5].replace("lr", ""))
        epochs = int(parts[6].replace("epochs", ""))
        
        with open(path, "r") as f:
            data = json.load(f)
            
        results.append({
            'head_align': head_align,
            'lr': lr,
            'epochs': epochs,
            'mnist': data['mnist'],
            'fmnist': data['fmnist'],
            'cifar10': data['cifar10'],
            'avg': data['avg']
        })
        
    print(f"Loaded {len(results)} configurations.")
    
    # We want to print tables for SFT and TTA separately
    for head in ["sft", "tta"]:
        subset = [r for r in results if r['head_align'] == head]
        if not subset:
            continue
            
        print(f"\n--- Hyperparameter Sweep Results for REDA-{head.upper()} (WA, Seed 42, N=128) ---")
        print(f"{'LR':<8} | {'Epochs':<6} | {'MNIST (%)':<10} | {'F-MNIST (%)':<12} | {'CIFAR-10 (%)':<13} | {'Mean (%)':<8}")
        print("-" * 65)
        
        # Sort by LR, then epochs
        subset = sorted(subset, key=lambda x: (x['lr'], x['epochs']))
        for r in subset:
            print(f"{r['lr']:<8g} | {r['epochs']:<6} | {r['mnist']:<10.2f} | {r['fmnist']:<12.2f} | {r['cifar10']:<13.2f} | {r['avg']:<8.2f}")

if __name__ == "__main__":
    main()
