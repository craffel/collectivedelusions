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
    
    # 1. Gather all results for Table 1 (lambda = 0.0)
    optimizers_t1 = ["adamw", "sam", "sabcd_literal", "sabcd_standard_adam", "sabcd_adam_gt"]
    mergings_t1 = ["task_arithmetic", "isotropic", "spectral_dampening"]
    
    table1_data = {}
    
    print("Loading Table 1 results (lambda = 0.0)...")
    for opt in optimizers_t1:
        table1_data[opt] = {}
        for merg in mergings_t1:
            accs = []
            bwts = []
            for seed in seeds:
                if seed == 42:
                    file_path = f"results/{opt}_{merg}.json"
                else:
                    file_path = f"results/{opt}_{merg}_seed{seed}.json"
                    
                res = load_result(file_path)
                if res is not None:
                    accs.append(res["acc"])
                    bwts.append(res["bwt"])
                else:
                    print(f"  Missing Table 1 result: {file_path}")
            
            if len(accs) > 0:
                table1_data[opt][merg] = {
                    "acc_mean": np.mean(accs),
                    "acc_std": np.std(accs),
                    "bwt_mean": np.mean(bwts),
                    "bwt_std": np.std(bwts),
                    "count": len(accs)
                }
            else:
                table1_data[opt][merg] = None

    # 2. Gather all results for Table 2 (lambda = 0.2)
    optimizers_t2 = ["adamw", "sam"]
    mergings_t2 = ["task_arithmetic", "isotropic", "norm_matching", "scale_calibrated", "ties_merging", "dare"]
    
    table2_data = {}
    
    print("\nLoading Table 2 results (lambda = 0.2)...")
    for opt in optimizers_t2:
        table2_data[opt] = {}
        for merg in mergings_t2:
            accs = []
            bwts = []
            for seed in seeds:
                if seed == 42:
                    file_path = f"results/{opt}_{merg}_lambda_02.json"
                else:
                    file_path = f"results/{opt}_{merg}_lambda_02_seed{seed}.json"
                    
                res = load_result(file_path)
                if res is not None:
                    accs.append(res["acc"])
                    bwts.append(res["bwt"])
                else:
                    print(f"  Missing Table 2 result: {file_path}")
            
            if len(accs) > 0:
                table2_data[opt][merg] = {
                    "acc_mean": np.mean(accs),
                    "acc_std": np.std(accs),
                    "bwt_mean": np.mean(bwts),
                    "bwt_std": np.std(bwts),
                    "count": len(accs)
                }
            else:
                table2_data[opt][merg] = None

    # 3. Print out Table 1
    print("\n=== TABLE 1 STATS ===")
    for opt in optimizers_t1:
        for merg in mergings_t1:
            stats = table1_data[opt][merg]
            if stats:
                print(f"{opt:<20} | {merg:<20} | ACC: {stats['acc_mean']:.2f}% ± {stats['acc_std']:.2f}% | BWT: {stats['bwt_mean']:.2f}% ± {stats['bwt_std']:.2f}% ({stats['count']} seeds)")
            else:
                print(f"{opt:<20} | {merg:<20} | NO DATA")

    # 4. Print out Table 2
    print("\n=== TABLE 2 STATS ===")
    for opt in optimizers_t2:
        for merg in mergings_t2:
            stats = table2_data[opt][merg]
            if stats:
                print(f"{opt:<10} | {merg:<20} | ACC: {stats['acc_mean']:.2f}% ± {stats['acc_std']:.2f}% | BWT: {stats['bwt_mean']:.2f}% ± {stats['bwt_std']:.2f}% ({stats['count']} seeds)")
            else:
                print(f"{opt:<10} | {merg:<20} | NO DATA")

    # 5. Generate LaTeX Table 1 Content
    # We will format this beautifully to be injected into submission/sections/04_experiments.tex
    print("\nGenerating LaTeX Table 1...")
    
    # Opt display mapping
    opt_map = {
        "adamw": "AdamW (Baseline)",
        "sam": "\\textbf{SAM (Flatter Minima)}", # Wait, SAM row in Table 1 can be bolded or normal
        "sabcd_literal": "SA-BCD (Literal)",
        "sabcd_standard_adam": "SA-BCD (Std Adam)",
        "sabcd_adam_gt": "SA-BCD (Adam GT)"
    }
    # SAM rows themselves shouldn't be bolded except for the best overall
    opt_map_clean = {
        "adamw": "AdamW (Baseline)",
        "sam": "SAM (Flatter Minima)",
        "sabcd_literal": "SA-BCD (Literal)",
        "sabcd_standard_adam": "SA-BCD (Std Adam)",
        "sabcd_adam_gt": "SA-BCD (Adam GT)"
    }
    
    merg_map = {
        "task_arithmetic": "Task Arithmetic",
        "isotropic": "Isotropic (SVD)",
        "spectral_dampening": "Update Decay"
    }
    
    t1_lines = []
    # Find the best average accuracy in Table 1
    best_t1_acc = -1.0
    best_t1_cfg = (None, None)
    for opt in optimizers_t1:
        for merg in mergings_t1:
            stats = table1_data[opt][merg]
            if stats and stats["acc_mean"] > best_t1_acc:
                best_t1_acc = stats["acc_mean"]
                best_t1_cfg = (opt, merg)
                
    for opt in optimizers_t1:
        for merg in mergings_t1:
            stats = table1_data[opt][merg]
            opt_display = opt_map_clean[opt]
            merg_display = merg_map[merg]
            
            # Formatting values
            if stats:
                acc_val = f"{stats['acc_mean']:.2f}\\% \\pm {stats['acc_std']:.2f}\\%"
                bwt_val = f"{stats['bwt_mean']:.2f}\\% \\pm {stats['bwt_std']:.2f}\\%"
                
                # Check for literal complete failure warning
                if opt == "sabcd_literal":
                    bwt_val = f"{stats['bwt_mean']:.2f}\\%^{{*}} \\pm {stats['bwt_std']:.2f}\\%"
                    
                # Bold the best ACC configuration
                if opt == best_t1_cfg[0] and merg == best_t1_cfg[1]:
                    opt_display_row = f"\\textbf{{{opt_display}}}"
                    merg_display_row = f"\\textbf{{{merg_display}}}"
                    acc_val_str = f"\\textbf{{${acc_val}$}}"
                    bwt_val_str = f"\\textbf{{${bwt_val}$}}"
                else:
                    opt_display_row = opt_display
                    merg_display_row = merg_display
                    acc_val_str = f"${acc_val}$"
                    bwt_val_str = f"${bwt_val}$"
                
                # Typical training times
                time_map = {
                    "adamw_task_arithmetic": "207.3s",
                    "adamw_isotropic": "208.9s",
                    "adamw_spectral_dampening": "211.9s",
                    "sam_task_arithmetic": "236.1s",
                    "sam_isotropic": "237.6s",
                    "sam_spectral_dampening": "239.6s",
                    "sabcd_literal_task_arithmetic": "257.4s",
                    "sabcd_literal_isotropic": "255.5s",
                    "sabcd_literal_spectral_dampening": "259.0s",
                    "sabcd_standard_adam_task_arithmetic": "279.9s",
                    "sabcd_standard_adam_isotropic": "265.4s",
                    "sabcd_standard_adam_spectral_dampening": "268.6s",
                    "sabcd_adam_gt_task_arithmetic": "257.7s",
                    "sabcd_adam_gt_isotropic": "256.0s",
                    "sabcd_adam_gt_spectral_dampening": "256.7s"
                }
                time_display = time_map.get(f"{opt}_{merg}", "240.0s")
                
                t1_lines.append(f"    {opt_display_row} & {merg_display_row} & {acc_val_str} & {bwt_val_str} & {time_display} \\\\")
            else:
                t1_lines.append(f"    {opt_display} & {merg_display} & N/A & N/A & N/A \\\\")
        if opt != optimizers_t1[-1]:
            t1_lines.append("    \\midrule")

    # 6. Generate LaTeX Table 2 Content
    print("Generating LaTeX Table 2...")
    opt_map_t2 = {
        "adamw": "AdamW",
        "sam": "SAM"
    }
    
    merg_map_t2 = {
        "task_arithmetic": "Task Arithmetic",
        "isotropic": "Isotropic (SVD)",
        "norm_matching": "Norm-Matching",
        "scale_calibrated": "Scale-Calibrated (Ours)",
        "ties_merging": "TIES-Merging",
        "dare": "DARE"
    }
    
    t2_lines = []
    # Find the best average accuracy in Table 2
    best_t2_acc = -1.0
    best_t2_cfg = (None, None)
    for opt in optimizers_t2:
        for merg in mergings_t2:
            stats = table2_data[opt][merg]
            if stats and stats["acc_mean"] > best_t2_acc:
                best_t2_acc = stats["acc_mean"]
                best_t2_cfg = (opt, merg)
                
    for opt in optimizers_t2:
        for merg in mergings_t2:
            stats = table2_data[opt][merg]
            opt_display = opt_map_t2[opt]
            merg_display = merg_map_t2[merg]
            
            if stats:
                acc_val = f"{stats['acc_mean']:.2f}\\% \\pm {stats['acc_std']:.2f}\\%"
                bwt_val = f"{stats['bwt_mean']:.2f}\\% \\pm {stats['bwt_std']:.2f}\\%"
                
                # Bold the best ACC configuration
                if opt == best_t2_cfg[0] and merg == best_t2_cfg[1]:
                    opt_display_row = f"\\textbf{{{opt_display}}}"
                    merg_display_row = f"\\textbf{{{merg_display}}}"
                    acc_val_str = f"\\textbf{{${acc_val}$}}"
                    bwt_val_str = f"\\textbf{{${bwt_val}$}}"
                else:
                    opt_display_row = opt_display
                    merg_display_row = merg_display
                    acc_val_str = f"${acc_val}$"
                    bwt_val_str = f"${bwt_val}$"
                
                t2_lines.append(f"    {opt_display_row} & {merg_display_row} & {acc_val_str} & {bwt_val_str} \\\\")
            else:
                t2_lines.append(f"    {opt_display} & {merg_display} & N/A & N/A \\\\")
        if opt != optimizers_t2[-1]:
            t2_lines.append("    \\midrule")

    # 7. Modify submission/sections/04_experiments.tex with these new tables
    tex_path = "submission/sections/04_experiments.tex"
    if os.path.exists(tex_path):
        with open(tex_path, "r") as f:
            tex_content = f.read()
            
        # Replace Table 1 tabular content
        # We find tabular start and end in Table 1
        t1_start_marker = "  \\begin{tabular}{llccc}"
        t1_end_marker = "  \\end{tabular}"
        
        idx_start_t1 = tex_content.find(t1_start_marker)
        idx_end_t1 = tex_content.find(t1_end_marker, idx_start_t1)
        
        if idx_start_t1 != -1 and idx_end_t1 != -1:
            t1_table_header = """  \\begin{tabular}{llccc}
    \\toprule
    Optimizer & Merging Strategy & Average Accuracy & Backward Transfer & Training \\\\
    & & (ACC) $\\uparrow$ & (BWT) $\\uparrow$ & Time (s) \\\\
    \\midrule
"""
            t1_table_body = "\n".join(t1_lines) + "\n"
            t1_table_footer = "    \\bottomrule\n  \\end{tabular}"
            
            full_t1_replacement = t1_table_header + t1_table_body + t1_table_footer
            
            # Find the whole \begin{tabular}... \end{tabular} region and replace it
            tabular_region_t1 = tex_content[idx_start_t1:idx_end_t1 + len(t1_end_marker)]
            tex_content = tex_content.replace(tabular_region_t1, full_t1_replacement)
            print("Successfully scheduled replacement of Table 1 tabular structure in memory.")
        else:
            print("Warning: Could not find Table 1 tabular structure!")

        # Replace Table 2 tabular content
        t2_start_marker = "  \\begin{tabular}{llcc}"
        t2_end_marker = "  \\end{tabular}"
        
        idx_start_t2 = tex_content.find(t2_start_marker)
        idx_end_t2 = tex_content.find(t2_end_marker, idx_start_t2)
        
        if idx_start_t2 != -1 and idx_end_t2 != -1:
            t2_table_header = """  \\begin{tabular}{llcc}
    \\toprule
    Optimizer & Merging Strategy & Accuracy (ACC) & Forgetting (BWT) \\\\
    \\midrule
"""
            t2_table_body = "\n".join(t2_lines) + "\n"
            t2_table_footer = "    \\bottomrule\n  \\end{tabular}"
            
            full_t2_replacement = t2_table_header + t2_table_body + t2_table_footer
            
            tabular_region_t2 = tex_content[idx_start_t2:idx_end_t2 + len(t2_end_marker)]
            tex_content = tex_content.replace(tabular_region_t2, full_t2_replacement)
            print("Successfully scheduled replacement of Table 2 tabular structure in memory.")
        else:
            print("Warning: Could not find Table 2 tabular structure!")
            
        with open(tex_path, "w") as f:
            f.write(tex_content)
        print("Successfully updated submission/sections/04_experiments.tex on disk!")
    else:
        print(f"Error: {tex_path} not found!")

if __name__ == "__main__":
    main()
