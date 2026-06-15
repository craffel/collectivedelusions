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
    
    # 1. Gather all results
    # Lambda = 0.0
    adamw_ta_00 = []
    adamw_iso_00 = []
    sam_ta_00 = []
    sam_iso_00 = []
    
    # Lambda = 0.2
    adamw_ta_02 = []
    adamw_iso_02 = []
    adamw_norm_02 = []
    sam_ta_02 = []
    sam_iso_02 = []
    sam_norm_02 = []
    
    for seed in seeds:
        # Lambda = 0.0 files
        f_adamw_ta_00 = f"results/adamw_task_arithmetic_seed{seed}.json" if seed != 42 else "results/adamw_task_arithmetic.json"
        f_adamw_iso_00 = f"results/adamw_isotropic_seed{seed}.json" if seed != 42 else "results/adamw_isotropic.json"
        f_sam_ta_00 = f"results/sam_task_arithmetic_seed{seed}.json" if seed != 42 else "results/sam_task_arithmetic.json"
        f_sam_iso_00 = f"results/sam_isotropic_seed{seed}.json" if seed != 42 else "results/sam_isotropic.json"
        
        # Lambda = 0.2 files
        f_adamw_ta_02 = f"results/adamw_task_arithmetic_lambda_02_seed{seed}.json" if seed != 42 else "results/adamw_task_arithmetic_lambda_02.json"
        f_adamw_iso_02 = f"results/adamw_isotropic_lambda_02_seed{seed}.json" if seed != 42 else "results/adamw_isotropic_lambda_02.json"
        f_adamw_norm_02 = f"results/adamw_norm_matching_lambda_02_seed{seed}.json" if seed != 42 else "results/adamw_norm_matching_lambda_02.json"
        f_sam_ta_02 = f"results/sam_task_arithmetic_lambda_02_seed{seed}.json" if seed != 42 else "results/sam_task_arithmetic_lambda_02.json"
        f_sam_iso_02 = f"results/sam_isotropic_lambda_02_seed{seed}.json" if seed != 42 else "results/sam_isotropic_lambda_02.json"
        f_sam_norm_02 = f"results/sam_norm_matching_lambda_02_seed{seed}.json" if seed != 42 else "results/sam_norm_matching_lambda_02.json"
        
        r = load_result(f_adamw_ta_00); r and adamw_ta_00.append(r)
        r = load_result(f_adamw_iso_00); r and adamw_iso_00.append(r)
        r = load_result(f_sam_ta_00); r and sam_ta_00.append(r)
        r = load_result(f_sam_iso_00); r and sam_iso_00.append(r)
        
        r = load_result(f_adamw_ta_02); r and adamw_ta_02.append(r)
        r = load_result(f_adamw_iso_02); r and adamw_iso_02.append(r)
        r = load_result(f_adamw_norm_02); r and adamw_norm_02.append(r)
        r = load_result(f_sam_ta_02); r and sam_ta_02.append(r)
        r = load_result(f_sam_iso_02); r and sam_iso_02.append(r)
        r = load_result(f_sam_norm_02); r and sam_norm_02.append(r)

    # Verify that all results are loaded
    expected_len = len(seeds)
    if len(adamw_ta_00) < expected_len or len(adamw_iso_00) < expected_len or len(sam_ta_00) < expected_len or len(sam_iso_00) < expected_len:
        print("Warning: Some lambda = 0.0 results are missing!")
        return
    if len(adamw_ta_02) < expected_len or len(adamw_iso_02) < expected_len or len(adamw_norm_02) < expected_len or len(sam_ta_02) < expected_len or len(sam_iso_02) < expected_len or len(sam_norm_02) < expected_len:
        print("Warning: Some lambda = 0.2 results are missing!")
        return

    print("All multi-seed results successfully loaded!")
    
    # Calculate means and std
    stats = {}
    
    def get_stats(lst):
        accs = [x["acc"] for x in lst]
        bwts = [x["bwt"] for x in lst]
        return {
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "bwt_mean": np.mean(bwts), "bwt_std": np.std(bwts)
        }
        
    stats["adamw_ta_00"] = get_stats(adamw_ta_00)
    stats["adamw_iso_00"] = get_stats(adamw_iso_00)
    stats["sam_ta_00"] = get_stats(sam_ta_00)
    stats["sam_iso_00"] = get_stats(sam_iso_00)
    
    stats["adamw_ta_02"] = get_stats(adamw_ta_02)
    stats["adamw_iso_02"] = get_stats(adamw_iso_02)
    stats["adamw_norm_02"] = get_stats(adamw_norm_02)
    stats["sam_ta_02"] = get_stats(sam_ta_02)
    stats["sam_iso_02"] = get_stats(sam_iso_02)
    stats["sam_norm_02"] = get_stats(sam_norm_02)

    # 2. Modify submission/sections/04_experiments.tex
    tex_path = "submission/sections/04_experiments.tex"
    if not os.path.exists(tex_path):
        print(f"Error: {tex_path} does not exist!")
        return
        
    with open(tex_path, "r") as f:
        content = f.read()
        
    # Table 1 Updates
    s = stats["adamw_ta_00"]
    old_row = "    AdamW (Baseline) & Task Arithmetic (Average) & 59.64\\% & -38.61\\% & 207.3s \\\\"
    new_row = f"    AdamW (Baseline) & Task Arithmetic (Average) & ${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$ & ${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$ & 207.3s \\\\"
    content = content.replace(old_row, new_row)
    
    s = stats["adamw_iso_00"]
    old_row = "    AdamW (Baseline) & Isotropic Merging (SVD) & 56.38\\% & -40.33\\% & 208.9s \\\\"
    new_row = f"    AdamW (Baseline) & Isotropic Merging (SVD) & ${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$ & ${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$ & 208.9s \\\\"
    content = content.replace(old_row, new_row)
    
    s = stats["sam_ta_00"]
    old_row = "    \\textbf{SAM (Flatter Minima)} & \\textbf{Task Arithmetic (Average)} & \\textbf{68.27\\%} & \\textbf{-29.64\\%} & 236.1s \\\\"
    new_row = f"    \\textbf{{SAM (Flatter Minima)}} & \\textbf{{Task Arithmetic (Average)}} & \\textbf{{${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$}} & \\textbf{{${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$}} & 236.1s \\\\"
    content = content.replace(old_row, new_row)
    
    s = stats["sam_iso_00"]
    old_row = "    SAM (Flatter Minima) & Isotropic Merging (SVD) & 62.46\\% & -35.18\\% & 237.6s \\\\"
    new_row = f"    SAM (Flatter Minima) & Isotropic Merging (SVD) & ${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$ & ${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$ & 237.6s \\\\"
    content = content.replace(old_row, new_row)
    
    # Table 2 Updates
    s = stats["adamw_ta_02"]
    old_row = "    AdamW & Task Arithmetic & 62.58\\% & -31.84\\% \\\\"
    new_row = f"    AdamW & Task Arithmetic & ${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$ & ${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$ \\\\"
    content = content.replace(old_row, new_row)
    
    s = stats["adamw_iso_02"]
    old_row = "    AdamW & Isotropic (SVD) & 70.98\\% & -18.86\\% \\\\"
    new_row = f"    AdamW & Isotropic (SVD) & ${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$ & ${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$ \\\\"
    content = content.replace(old_row, new_row)
    
    s = stats["adamw_norm_02"]
    old_row = "    AdamW & Norm-Matching & 52.31\\% & -39.61\\% \\\\"
    new_row = f"    AdamW & Norm-Matching & ${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$ & ${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$ \\\\"
    content = content.replace(old_row, new_row)
    
    s = stats["sam_ta_02"]
    old_row = "    SAM & Task Arithmetic & 73.52\\% & -20.94\\% \\\\"
    new_row = f"    SAM & Task Arithmetic & ${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$ & ${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$ \\\\"
    content = content.replace(old_row, new_row)
    
    s = stats["sam_iso_02"]
    old_row = "    SAM & Isotropic (SVD) & \\textbf{76.00\\%} & \\textbf{-14.84\\%} \\\\"
    new_row = f"    SAM & Isotropic (SVD) & \\textbf{{${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$}} & \\textbf{{${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$}} \\\\"
    content = content.replace(old_row, new_row)
    
    s = stats["sam_norm_02"]
    old_row = "    SAM & Norm-Matching & 54.81\\% & -38.90\\% \\\\"
    new_row = f"    SAM & Norm-Matching & ${s['acc_mean']:.2f}\\% \\pm {s['acc_std']:.2f}\\%$ & ${s['bwt_mean']:.2f}\\% \\pm {s['bwt_std']:.2f}\\%$ \\\\"
    content = content.replace(old_row, new_row)
    
    with open(tex_path, "w") as f:
        f.write(content)
        
    print("Surgically updated submission/sections/04_experiments.tex with multi-seed statistics!")

if __name__ == "__main__":
    main()
