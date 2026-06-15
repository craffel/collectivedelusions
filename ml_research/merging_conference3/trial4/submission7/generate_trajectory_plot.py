import os
import numpy as np
import matplotlib.pyplot as plt
from run_experiments import (
    generate_landscape_parameters,
    run_online_adamerging,
    run_online_polymerge,
    run_ofs_tune,
    run_ofs_unconstrained,
    L, SUITES
)

def main():
    print("Generating Coefficient Trajectories Plot...")
    seed = 42
    suite_id = "Suite B"
    suite_info = SUITES[suite_id]
    tasks = suite_info["tasks"] # [2, 3] for CIFAR-10 and SVHN
    init_val = suite_info["init_val"]
    
    # Generate parameters for seed 42
    A, B, alpha_opt = generate_landscape_parameters(seed)
    
    # Generate validation profile as in run_experiments.py
    np.random.seed(seed + 1000 + len(tasks))
    eps_val = np.random.normal(0, 0.10, size=(4, L))
    v_bias = np.random.normal(0, 0.03, size=(4, 1))
    alpha_val_opt = np.clip(alpha_opt + eps_val + v_bias, 0.0, 1.0)
    
    # Run the optimizations to get alpha profiles
    print("Running Online AdaMerging...")
    alpha_ada = run_online_adamerging(tasks, A, B, alpha_opt, init_val)
    
    print("Running Online PolyMerge...")
    alpha_poly = run_online_polymerge(tasks, A, B, alpha_opt, init_val)
    
    print("Running Offline OFS-Tune...")
    alpha_ofs = run_ofs_tune(tasks, A, B, alpha_val_opt, init_val)
    
    print("Running Offline OFS-Unconstrained...")
    alpha_uncon = run_ofs_unconstrained(tasks, A, B, alpha_val_opt, init_val)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    layers = np.arange(1, L + 1)
    
    colors = {
        "True Optimal": "#000000",
        "Uniform": "#888888",
        "AdaMerging": "#E74C3C",
        "PolyMerge": "#E67E22",
        "OFS-Tune": "#2ECC71",
        "OFS-Unconstrained": "#3498DB"
    }
    
    # We have 2 tasks in Suite B: tasks[0]=2 (CIFAR-10) and tasks[1]=3 (SVHN)
    task_names = {2: "CIFAR-10", 3: "SVHN"}
    
    for idx, t_idx in enumerate(tasks):
        ax = axes[idx]
        
        # True Optimal
        ax.plot(layers, alpha_opt[t_idx], label="Ground Truth Optimal", color=colors["True Optimal"], linestyle="--", linewidth=2.5, zorder=5)
        
        # Uniform
        ax.axhline(init_val, label="Uniform (Baseline)", color=colors["Uniform"], linestyle=":", linewidth=2, zorder=1)
        
        # Online AdaMerging
        ax.plot(layers, alpha_ada[idx], label="Online AdaMerging (TTA)", color=colors["AdaMerging"], marker="o", linewidth=1.8, alpha=0.85, zorder=3)
        
        # Online PolyMerge
        ax.plot(layers, alpha_poly[idx], label="Online PolyMerge (TTA, d=2)", color=colors["PolyMerge"], marker="^", linewidth=1.8, alpha=0.85, zorder=4)
        
        # Offline OFS-Unconstrained
        ax.plot(layers, alpha_uncon[idx], label="Offline OFS-Unconstrained", color=colors["OFS-Unconstrained"], marker="s", linewidth=1.5, linestyle="-.", alpha=0.7, zorder=2)
        
        # Offline OFS-Tune (Ours)
        ax.plot(layers, alpha_ofs[idx], label="Offline OFS-Tune (Ours, d=1)", color=colors["OFS-Tune"], marker="D", linewidth=2.2, alpha=0.9, zorder=5)
        
        ax.set_title(f"Task: {task_names[t_idx]} (Suite B)", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Layer Index (Depth)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Merging Coefficient Value (\\alpha)", fontsize=11)
        ax.set_xticks(layers)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        
    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10.5, bbox_to_anchor=(0.5, -0.04))
    
    plt.suptitle("Optimized Merging Coefficient Trajectories across Layers (Suite B, Seed 42)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    # Adjust subplot rect to make room for legend at the bottom
    plt.subplots_adjust(bottom=0.15)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("submission", exist_ok=True)
    plt.savefig("results/coefficient_trajectories.png", dpi=300, bbox_inches='tight')
    plt.savefig("submission/coefficient_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Coefficient trajectories plot successfully saved!")

if __name__ == "__main__":
    main()
