import os
import json
import numpy as np

def main():
    results_dir = "./results"
    files = os.listdir(results_dir)
    
    # We want to group by configuration: merge_method, rep_cal, head_align, cal_size
    # Format of filename: {merge}_{lam}_{rep}_{head}_{cal_size}_seed{seed}.json
    # Examples:
    # wa_lam0.0_repnone_headnone_N128_seed42.json
    # ta_lam0.3_replsc_headnone_N128_seed100.json
    
    configs = {}
    
    for filename in files:
        if not filename.endswith(".json"):
            continue
        
        path = os.path.join(results_dir, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        # Parse filename
        parts = filename[:-5].split("_")
        # wa_lam0.0_repnone_headnone_N128_seed42
        merge = parts[0]
        # parts[1] is 'lam0.0' or 'lam0.3'
        lam = parts[1]
        rep = parts[2][3:] # strip 'rep'
        head = parts[3][4:] # strip 'head'
        
        # parts[4] is 'N128' or similar
        n_size = parts[4]
        
        # parts[5] is 'seed42'
        seed = parts[5][4:]
        
        config_key = (merge, lam, rep, head, n_size)
        if config_key not in configs:
            configs[config_key] = []
        configs[config_key].append(data)
        
    print("Parsed", len(configs), "configurations.")
    
    # We want to summarize N128 configurations
    n128_configs = {k: v for k, v in configs.items() if k[4] == "N128"}
    
    # Print LaTeX table rows
    print("\n\\begin{tabular}{lllcccc}")
    print("  \\toprule")
    print("  Merge & Rep. Cal. & Head Align. & MNIST & F-MNIST & CIFAR-10 & Mean \\\\")
    print("  \\midrule")
    
    # Let's sort the rows nicely: first WA, then TA. Within each, NONE, LSC, TSC, NTAAC, etc.
    wa_rows = [
        ("none", "none"),
        ("lsc", "none"),
        ("tsc", "none"),
        ("ntaac", "none"),
        ("none", "sft"),
        ("none", "tta"),
        ("tsc", "tta"),
        ("lsc", "tta"),
        ("ntaac", "sft"),
        ("ntaac", "tta"),
    ]
    
    ta_rows = [
        ("none", "none"),
        ("ntaac", "none"),
        ("none", "tta"),
        ("ntaac", "tta"),
    ]
    
    def print_rows(merge_method, lam_val, rows_to_print):
        for rep, head in rows_to_print:
            key = (merge_method, lam_val, rep, head, "N128")
            if key not in configs:
                print(f"  % Missing key: {key}")
                continue
            runs = configs[key]
            
            mnist_vals = [run["mnist"] for run in runs]
            fmnist_vals = [run["fmnist"] for run in runs]
            cifar_vals = [run["cifar10"] for run in runs]
            avg_vals = [run["avg"] for run in runs]
            
            mnist_mean, mnist_std = np.mean(mnist_vals), np.std(mnist_vals)
            fmnist_mean, fmnist_std = np.mean(fmnist_vals), np.std(fmnist_vals)
            cifar_mean, cifar_std = np.mean(cifar_vals), np.std(cifar_vals)
            avg_mean, avg_std = np.mean(avg_vals), np.std(avg_vals)
            
            merge_name = merge_method.upper()
            rep_name = rep.upper() if rep != "none" else "NONE"
            head_name = head.upper() if head != "none" else "NONE"
            
            print(f"  {merge_name} & {rep_name} & {head_name} & "
                  f"{mnist_mean:.2f} (\\pm{mnist_std:.2f}) & "
                  f"{fmnist_mean:.2f} (\\pm{fmnist_std:.2f}) & "
                  f"{cifar_mean:.2f} (\\pm{cifar_std:.2f}) & "
                  f"{avg_mean:.2f} (\\pm{avg_std:.2f}) \\\\")
                  
    print("  % WA Rows")
    print_rows("wa", "lam0.0", wa_rows)
    print("  \\midrule")
    print("  % TA Rows")
    print_rows("ta", "lam0.3", ta_rows)
    print("  \\bottomrule")
    print("\\end{tabular}")
    
    # Also print the sample efficiency numbers for wa with seed 42
    print("\n--- SAMPLE EFFICIENCY DATA (WA, Seed 42) ---")
    # We want to print for repnone_headsft, repnone_headtta, repntaac_headnone, repntaac_headsft, repntaac_headtta
    sizes = ["N16", "N32", "N64", "N128", "N256", "N512"]
    variants = [
        ("none", "sft"),
        ("none", "tta"),
        ("ntaac", "none"),
        ("ntaac", "sft"),
        ("ntaac", "tta")
    ]
    
    for rep, head in variants:
        print(f"Variant: {rep.upper()} + {head.upper()}")
        for sz in sizes:
            # find the file with seed42
            filename = f"wa_lam0.0_rep{rep}_head{head}_{sz}_seed42.json"
            path = os.path.join(results_dir, filename)
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                print(f"  {sz}: MNIST={data['mnist']:.2f}, F-MNIST={data['fmnist']:.2f}, CIFAR={data['cifar10']:.2f}, AVG={data['avg']:.2f}")
            else:
                print(f"  {sz}: Missing")

if __name__ == "__main__":
    main()
