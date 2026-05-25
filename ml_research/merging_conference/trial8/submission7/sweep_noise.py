import os
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    noise_levels = [0.0, 0.3, 0.6, 0.9]
    modes = ["none", "scts_euclidean", "scts_cosine", "entropy_euclidean", "entropy_cosine", "cpal_euclidean", "cpal_cosine"]
    
    # Store results
    # Structure: {mode: {noise_level: {"mnist": acc, "fashion": acc}}}
    sweep_results = {m: {} for m in modes}
    
    print("Starting noise sweep...")
    for lvl in noise_levels:
        out_dir = f"results_sweep_{lvl}"
        print(f"\n>>> Running experiment for noise_std = {lvl} in directory {out_dir}...", flush=True)
        
        # Run experiment.py as a subprocess with output redirected to devnull to avoid buffering issues
        cmd = [
            "python3", "experiment.py",
            "--noise_std", str(lvl),
            "--steps", "10",
            "--batch_size", "128",
            "--out_dir", out_dir
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error running for noise level {lvl}:")
            print(result.stderr)
            return
            
        print(f"Finished noise_std = {lvl}.", flush=True)
            
        # Read the generated results.json
        res_json_path = os.path.join(out_dir, "results.json")
        with open(res_json_path, "r") as f:
            res_data = json.load(f)
            
        # Extract Noisy MNIST and Noisy Fashion accuracies
        for m in modes:
            mnist_acc = res_data[m]["Noisy MNIST"]["acc"]
            fashion_acc = res_data[m]["Noisy Fashion"]["acc"]
            sweep_results[m][lvl] = {
                "mnist": mnist_acc,
                "fashion": fashion_acc
            }
            
    print("\nSweep completed! Results summary:")
    # Print Markdown table
    print("\n| Mode | Noise 0.0 | Noise 0.3 | Noise 0.6 | Noise 0.9 |")
    print("|---|---|---|---|---|")
    for m in modes:
        row = f"| {m.upper()} | " + " | ".join([f"{sweep_results[m][lvl]['mnist']:.2f}%" for lvl in noise_levels]) + " |"
        print(row)
        
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mode label mapping for publication
    mode_labels = {
        "none": "Static (Uniform)",
        "scts_euclidean": "SCTS-E",
        "scts_cosine": "SCTS-C",
        "entropy_euclidean": "Entr-E",
        "entropy_cosine": "Entr-C",
        "cpal_euclidean": "CPAL-E",
        "cpal_cosine": "CPAL-C (Ours)"
    }
    
    markers = {
        "none": "o",
        "scts_euclidean": "s",
        "scts_cosine": "s",
        "entropy_euclidean": "D",
        "entropy_cosine": "D",
        "cpal_euclidean": "^",
        "cpal_cosine": "^"
    }
    
    linestyles = {
        "none": ":",
        "scts_euclidean": "--",
        "scts_cosine": "-",
        "entropy_euclidean": "--",
        "entropy_cosine": "-",
        "cpal_euclidean": "--",
        "cpal_cosine": "-"
    }
    
    colors = {
        "none": "gray",
        "scts_euclidean": "blue",
        "scts_cosine": "cyan",
        "entropy_euclidean": "green",
        "entropy_cosine": "limegreen",
        "cpal_euclidean": "red",
        "cpal_cosine": "darkred"
    }
    
    # Plot MNIST
    for m in modes:
        accs = [sweep_results[m][lvl]["mnist"] for lvl in noise_levels]
        ax1.plot(noise_levels, accs, label=mode_labels[m], marker=markers[m], 
                 linestyle=linestyles[m], color=colors[m], linewidth=2 if "cosine" in m or "Ours" in m else 1.5)
                 
    ax1.set_title("Noisy MNIST Accuracy vs. Input Noise Level", fontsize=12)
    ax1.set_xlabel("Noise Standard Deviation ($\sigma_n$)", fontsize=11)
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 105)
    
    # Plot Fashion
    for m in modes:
        accs = [sweep_results[m][lvl]["fashion"] for lvl in noise_levels]
        ax2.plot(noise_levels, accs, label=mode_labels[m], marker=markers[m], 
                 linestyle=linestyles[m], color=colors[m], linewidth=2 if "cosine" in m or "Ours" in m else 1.5)
                 
    ax2.set_title("Noisy Fashion Accuracy vs. Input Noise Level", fontsize=12)
    ax2.set_xlabel("Noise Standard Deviation ($\sigma_n$)", fontsize=11)
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower left", fontsize=9)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/noise_sweep.png", dpi=300)
    plt.close()
    print("\nSaved noise sweep plot to results/noise_sweep.png")
    
    # Save sweep results to json
    with open("results/sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=4)
        
if __name__ == "__main__":
    main()
