import os
import json
import numpy as np

def load_result(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return {
                "acc": data.get("acc"),
                "bwt": data.get("bwt")
            }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    seeds = [42, 43, 44]
    
    # Define the 10 configurations to check
    configs = [
        # Lambda = 0.0
        {"optimizer": "adamw", "merging": "task_arithmetic", "lambda_val": 0.0},
        {"optimizer": "adamw", "merging": "isotropic", "lambda_val": 0.0},
        {"optimizer": "sam", "merging": "task_arithmetic", "lambda_val": 0.0},
        {"optimizer": "sam", "merging": "isotropic", "lambda_val": 0.0},
        # Lambda = 0.2
        {"optimizer": "adamw", "merging": "task_arithmetic", "lambda_val": 0.2},
        {"optimizer": "adamw", "merging": "isotropic", "lambda_val": 0.2},
        {"optimizer": "sam", "merging": "task_arithmetic", "lambda_val": 0.2},
        {"optimizer": "sam", "merging": "isotropic", "lambda_val": 0.2},
        {"optimizer": "adamw", "merging": "norm_matching", "lambda_val": 0.2},
        {"optimizer": "sam", "merging": "norm_matching", "lambda_val": 0.2},
    ]
    
    print("Checking status of multi-seed jobs...")
    
    completed_configs = []
    missing_count = 0
    
    for run in configs:
        opt = run["optimizer"]
        merg = run["merging"]
        lamb = run["lambda_val"]
        
        accs = []
        bwts = []
        
        for seed in seeds:
            if lamb == 0.0:
                if seed == 42:
                    file_path = f"results/{opt}_{merg}.json"
                else:
                    file_path = f"results/{opt}_{merg}_seed{seed}.json"
            else:
                if seed == 42:
                    file_path = f"results/{opt}_{merg}_lambda_02.json"
                else:
                    file_path = f"results/{opt}_{merg}_lambda_02_seed{seed}.json"
                    
            res = load_result(file_path)
            if res is not None:
                accs.append(res["acc"])
                bwts.append(res["bwt"])
            else:
                missing_count += 1
                
        if len(accs) == len(seeds):
            completed_configs.append({
                "optimizer": opt,
                "merging": merg,
                "lambda_val": lamb,
                "acc_mean": np.mean(accs),
                "acc_std": np.std(accs),
                "bwt_mean": np.mean(bwts),
                "bwt_std": np.std(bwts),
                "acc_values": accs,
                "bwt_values": bwts
            })
            
    print(f"Completed configurations: {len(completed_configs)}/{len(configs)}")
    print(f"Missing results across all configurations and seeds: {missing_count}")
    
    if len(completed_configs) > 0:
        print("\n=== AGGREGATED MULTI-SEED RESULTS ===")
        print(f"{'Optimizer':<12} | {'Merging':<16} | {'Lambda':<6} | {'Accuracy %':<16} | {'Forgetting %':<16}")
        print("-" * 75)
        for c in completed_configs:
            acc_str = f"{c['acc_mean']:.2f}% ± {c['acc_std']:.2f}%"
            bwt_str = f"{c['bwt_mean']:.2f}% ± {c['bwt_std']:.2f}%"
            print(f"{c['optimizer']:<12} | {c['merging']:<16} | {c['lambda_val']:<6} | {acc_str:<16} | {bwt_str:<16}")
            
        print("\n=== LATEX FRAGMENTS ===")
        print("\n--- Table 1 (Lambda = 0.0) ---")
        for c in completed_configs:
            if c["lambda_val"] == 0.0:
                print(f"{c['optimizer']} & {c['merging']} & ${c['acc_mean']:.2f}\\% \\pm {c['acc_std']:.2f}\\%$ & ${c['bwt_mean']:.2f}\\% \\pm {c['bwt_std']:.2f}\\%$ \\\\")
                
        print("\n--- Table 2 (Lambda = 0.2) ---")
        for c in completed_configs:
            if c["lambda_val"] == 0.2:
                print(f"{c['optimizer']} & {c['merging']} & ${c['acc_mean']:.2f}\\% \\pm {c['acc_std']:.2f}\\%$ & ${c['bwt_mean']:.2f}\\% \\pm {c['bwt_std']:.2f}\\%$ \\\\")

if __name__ == "__main__":
    main()
