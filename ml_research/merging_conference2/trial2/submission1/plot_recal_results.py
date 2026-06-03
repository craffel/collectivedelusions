import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

def generate_plots():
    print("Gathering recalibrated results files...")
    files = glob.glob("results_recal_size*_layer*_seed*.json")
    if not files:
        if os.path.exists("results_recal_default.json"):
            files = ["results_recal_default.json"]
        else:
            print("No recalibrated results files found yet.")
            return

    # Structure to hold aggregated results
    opt_lambdas_by_config = {}
    opt_accs_by_config = {}
    
    sample_data = None
    configs = set()
    metrics = ["cka", "mse", "cosine", "mmd"]
    
    for f in files:
        try:
            with open(f, "r") as fh:
                res = json.load(fh)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
            
        size = res["cal_size"]
        layer = res["target_layer"]
        seed = res["seed"]
        config_key = (size, layer)
        configs.add(config_key)
        
        # Save sample data (e.g., size 128, layer4, seed 42)
        if sample_data is None or (size == 128 and layer == "layer4" and seed == 42):
            sample_data = res
            
        if config_key not in opt_lambdas_by_config:
            opt_lambdas_by_config[config_key] = {m: [] for m in metrics}
            opt_accs_by_config[config_key] = {m: [] for m in metrics}
            opt_lambdas_by_config[config_key]["oracle"] = []
            opt_lambdas_by_config[config_key]["oracle_acc"] = []
            opt_lambdas_by_config[config_key]["wa_acc"] = []
            
        for m in metrics:
            opt_lambdas_by_config[config_key][m].append(res["opt_lambdas"][m])
            opt_accs_by_config[config_key][m].append(res["opt_lambda_accs"][m])
            
        opt_lambdas_by_config[config_key]["oracle"].append(res["oracle_lambda"])
        opt_lambdas_by_config[config_key]["oracle_acc"].append(res["oracle_acc"])
        opt_lambdas_by_config[config_key]["wa_acc"].append(res["wa_avg_acc"])

    print(f"Processed {len(files)} files.")
    
    # ------------------ Plot 1: Oracle vs. AOS Alignment (Sample Run) ------------------
    if sample_data:
        print("Generating Plot 1: Oracle vs. AOS Alignment (Recalibrated)...")
        lambdas = np.array(sample_data["lambdas"])
        avg_test_accs = np.array(sample_data["avg_test_accs"])
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        color = "tab:red"
        ax1.set_xlabel(r"Weight Scaling Factor $\lambda$ (Task Arithmetic)", fontsize=12)
        ax1.set_ylabel("Average Test Accuracy (%)", color=color, fontsize=12)
        line_oracle, = ax1.plot(lambdas, avg_test_accs, color=color, linewidth=3, label="Average Test Accuracy (Oracle)")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("AOS Manifold Distance (Normalized)", color="tab:blue", fontsize=12)
        
        colors_dist = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
        lines_dist = []
        for idx, m in enumerate(metrics):
            dist_vals = np.array(sample_data["distances"][m])
            min_v, max_v = dist_vals.min(), dist_vals.max()
            if max_v > min_v:
                norm_dist = (dist_vals - min_v) / (max_v - min_v)
            else:
                norm_dist = dist_vals
                
            line_d, = ax2.plot(lambdas, norm_dist, color=colors_dist[idx], linestyle="--", alpha=0.8, linewidth=2,
                              label=f"AOS Dist ({m.upper()})")
            lines_dist.append(line_d)
            
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        
        lines = [line_oracle] + lines_dist
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower center", fontsize=10, bbox_to_anchor=(0.5, -0.25), ncol=3)
        
        plt.title(f"AOS-Recal Manifold Distance vs. Actual Test Accuracy\n(Calibration Size N={sample_data['cal_size']}, Layer={sample_data['target_layer']})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("aos_recal_alignment.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("aos_recal_alignment.png saved.")

    # ------------------ Plot 2: Performance vs. Calibration Set Size ------------------
    sizes = sorted(list(set([sz for sz, ly in configs])))
    layers = list(set([ly for sz, ly in configs]))
    
    if sizes:
        print("Generating Plot 2: Performance vs. Calibration Set Size (Recalibrated)...")
        fig, axes = plt.subplots(1, len(layers) if len(layers) > 0 else 1, figsize=(6 * len(layers), 5), sharey=False)
        if len(layers) == 1:
            axes = [axes]
            
        for l_idx, layer in enumerate(sorted(layers)):
            ax = axes[l_idx]
            
            all_oracles = []
            all_was = []
            for sz in sizes:
                if (sz, layer) in opt_lambdas_by_config:
                    all_oracles.extend(opt_lambdas_by_config[(sz, layer)]["oracle_acc"])
                    all_was.extend(opt_lambdas_by_config[(sz, layer)]["wa_acc"])
            mean_oracle = np.mean(all_oracles) if all_oracles else 0
            mean_wa = np.mean(all_was) if all_was else 0
            
            # Since recalibration makes Oracle dynamic depending on the calibration set (because stats are recalculated on N),
            # let's plot the average Oracle Acc for each size instead of a single horizontal line!
            oracle_means_by_size = []
            for sz in sizes:
                config_key = (sz, layer)
                if config_key in opt_lambdas_by_config:
                    oracle_means_by_size.append(np.mean(opt_lambdas_by_config[config_key]["oracle_acc"]))
                else:
                    oracle_means_by_size.append(0)
            
            ax.plot(sizes, oracle_means_by_size, color="black", linestyle="--", marker="x", linewidth=2, label="Oracle Optimal (Recal)")
            
            for m_idx, m in enumerate(metrics):
                means = []
                stds = []
                for sz in sizes:
                    config_key = (sz, layer)
                    if config_key in opt_accs_by_config:
                        vals = opt_accs_by_config[config_key][m]
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))
                    else:
                        means.append(0)
                        stds.append(0)
                
                ax.errorbar(sizes, means, yerr=stds, fmt="-o", color=colors_dist[m_idx], linewidth=2, capsize=4,
                            label=f"AOS ({m.upper()})")
                
            ax.set_title(f"Target Layer: {layer}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Calibration Set Size N", fontsize=11)
            if l_idx == 0:
                ax.set_ylabel("Average Merged Test Accuracy (%)", fontsize=11)
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.grid(True, linestyle="--", alpha=0.5)
            
        plt.suptitle("Model Merging Performance with BN Recalibration vs. Calibration Size", fontsize=14, fontweight="bold")
        axes[0].legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig("aos_recal_performance_vs_size.png", dpi=300)
        plt.close()
        print("aos_recal_performance_vs_size.png saved.")
        
    # Write summary
    summary = {}
    for config_key, d in opt_accs_by_config.items():
        size, layer = config_key
        config_name = f"size{size}_layer{layer}"
        summary[config_name] = {}
        for m in metrics:
            summary[config_name][m] = {
                "mean_acc": float(np.mean(d[m])),
                "std_acc": float(np.std(d[m])),
                "mean_lambda": float(np.mean(opt_lambdas_by_config[config_key][m])),
                "std_lambda": float(np.std(opt_lambdas_by_config[config_key][m]))
            }
        summary[config_name]["oracle"] = {
            "mean_acc": float(np.mean(opt_lambdas_by_config[config_key]["oracle_acc"])),
            "std_acc": float(np.std(opt_lambdas_by_config[config_key]["oracle_acc"])),
            "mean_lambda": float(np.mean(opt_lambdas_by_config[config_key]["oracle"])),
            "std_lambda": float(np.std(opt_lambdas_by_config[config_key]["oracle"]))
        }
        summary[config_name]["wa"] = {
            "mean_acc": float(np.mean(opt_lambdas_by_config[config_key]["wa_acc"])),
            "std_acc": float(np.std(opt_lambdas_by_config[config_key]["wa_acc"]))
        }
        
    with open("results_recal_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("results_recal_summary.json saved.")

if __name__ == "__main__":
    generate_plots()
