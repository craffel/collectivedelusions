import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random_seed = seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

def main():
    set_seed(42)
    print("Starting Transformer model-merging simulation...")
    
    # Define transformer structure (12 layers)
    num_layers = 12
    param_shapes = {}
    
    # Standard BERT-Tiny or ViT-Tiny style dimensions
    hidden_dim = 192
    mlp_dim = 768
    
    for l in range(num_layers):
        prefix = f"layer.{l}."
        param_shapes[prefix + "attn.q_proj.weight"] = (hidden_dim, hidden_dim)
        param_shapes[prefix + "attn.k_proj.weight"] = (hidden_dim, hidden_dim)
        param_shapes[prefix + "attn.v_proj.weight"] = (hidden_dim, hidden_dim)
        param_shapes[prefix + "attn.o_proj.weight"] = (hidden_dim, hidden_dim)
        param_shapes[prefix + "mlp.fc1.weight"] = (mlp_dim, hidden_dim)
        param_shapes[prefix + "mlp.fc2.weight"] = (hidden_dim, mlp_dim)
        
    print(f"Simulating {len(param_shapes)} tensors across {num_layers} layers.")
    
    # We will simulate two tasks with different layer-wise specialization profiles
    # Task 1: Deep Specialization (common in LLM fine-tuning where deeper layers adapt most)
    # Task 2: Middle Specialization (representing semantic adjustments concentrated in middle layers)
    
    task_vectors = {"task1": {}, "task2": {}}
    true_signals = {"task1": {}, "task2": {}}
    
    for task_name in ["task1", "task2"]:
        for name, shape in param_shapes.items():
            parts = name.split(".")
            layer_idx = int(parts[1])
            
            # Define specialized scale based on task
            if task_name == "task1":
                # Deeper layers have exponentially higher adaptation magnitude
                # (early layers are near zero, deeper layers are highly active)
                specialized_scale = 0.02 * ((layer_idx) / (num_layers - 1)) ** 4
            else:
                # Middle layers are highly active, early and late layers are near-zero
                specialized_scale = 0.02 * np.sin(np.pi * layer_idx / (num_layers - 1)) ** 4
                
            # Simulate true task-specific updates (signal)
            # A sparse signal where 15% of weights are actively adapted
            signal_mask = (torch.rand(shape) < 0.15).float()
            signal = torch.randn(shape) * specialized_scale * signal_mask
            
            # Simulate small background adaptation noise (low-magnitude updates)
            noise = torch.randn(shape) * 0.0001
            
            # Total task vector is signal + noise
            tv = signal + noise
            
            task_vectors[task_name][name] = tv
            true_signals[task_name][name] = (torch.abs(signal) > 0).float()
            
    print("Successfully simulated task vectors and ground-truth signal parameters.")
    
    # Evaluate GQ vs LQ masking at keep-ratio k = 0.20
    k_target = 0.20
    results = {}
    
    for task_name in ["task1", "task2"]:
        results[task_name] = {"GQ": {}, "LQ": {}}
        tvs = task_vectors[task_name]
        signals = true_signals[task_name]
        
        # --- Global Quantile (GQ) Masking ---
        all_vals = torch.cat([v.flatten() for v in tvs.values()])
        num_keep = int(k_target * len(all_vals))
        global_threshold = torch.topk(torch.abs(all_vals), num_keep).values[-1].item()
        
        gq_layer_keep_ratios = []
        gq_signal_retained = 0
        total_signals = 0
        
        layer_gq_counts = {l: {"kept": 0, "total": 0} for l in range(num_layers)}
        
        for name, tv in tvs.items():
            layer_idx = int(name.split(".")[1])
            mask = torch.abs(tv) >= global_threshold
            
            layer_gq_counts[layer_idx]["kept"] += mask.sum().item()
            layer_gq_counts[layer_idx]["total"] += tv.numel()
            
            sig_mask = signals[name]
            gq_signal_retained += (mask.float() * sig_mask).sum().item()
            total_signals += sig_mask.sum().item()
            
        gq_layer_ratios = [layer_gq_counts[l]["kept"] / layer_gq_counts[l]["total"] for l in range(num_layers)]
        gq_sig_preservation = gq_signal_retained / total_signals if total_signals > 0 else 0.0
        
        results[task_name]["GQ"]["layer_keep_ratios"] = gq_layer_ratios
        results[task_name]["GQ"]["signal_preservation"] = gq_sig_preservation
        results[task_name]["GQ"]["global_threshold"] = global_threshold
        
        # --- Layer-wise Quantile (LQ) Masking ---
        lq_signal_retained = 0
        layer_lq_counts = {l: {"kept": 0, "total": 0} for l in range(num_layers)}
        
        for name, tv in tvs.items():
            layer_idx = int(name.split(".")[1])
            flat_tv = tv.flatten()
            num_layer_keep = int(k_target * len(flat_tv))
            layer_threshold = torch.topk(torch.abs(flat_tv), num_layer_keep).values[-1].item()
            
            mask = torch.abs(tv) >= layer_threshold
            layer_lq_counts[layer_idx]["kept"] += mask.sum().item()
            layer_lq_counts[layer_idx]["total"] += tv.numel()
            
            sig_mask = signals[name]
            lq_signal_retained += (mask.float() * sig_mask).sum().item()
            
        lq_layer_ratios = [layer_lq_counts[l]["kept"] / layer_lq_counts[l]["total"] for l in range(num_layers)]
        lq_sig_preservation = lq_signal_retained / total_signals if total_signals > 0 else 0.0
        
        results[task_name]["LQ"]["layer_keep_ratios"] = lq_layer_ratios
        results[task_name]["LQ"]["signal_preservation"] = lq_sig_preservation
        
        print(f"\nResults for {task_name.upper()}:")
        print(f"  GQ Global Threshold: {global_threshold:.6f}")
        print(f"  GQ Ground-Truth Signal Preservation: {gq_sig_preservation * 100:.2f}%")
        print(f"  LQ Ground-Truth Signal Preservation: {lq_sig_preservation * 100:.2f}%")
        print("  GQ Layer-wise Keep Ratios:")
        for l, ratio in enumerate(gq_layer_ratios):
            print(f"    Layer {l:2d}: {ratio * 100:5.2f}% (vs. LQ {k_target*100:.2f}%)")
            
    # Save metrics to JSON
    os.makedirs("./results", exist_ok=True)
    with open("./results/nlp_simulation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate visualization plot
    plt.figure(figsize=(12, 5))
    
    # Plot Task 1 Layer-wise Keep Ratios
    plt.subplot(1, 2, 1)
    layers = np.arange(num_layers)
    plt.bar(layers - 0.2, [r * 100 for r in results["task1"]["GQ"]["layer_keep_ratios"]], width=0.4, label="GQ (Ours)", color="royalblue")
    plt.bar(layers + 0.2, [k_target * 100] * num_layers, width=0.4, label="LQ (Rigid)", color="darkorange", alpha=0.8)
    plt.xlabel("Transformer Layer Index")
    plt.ylabel("Parameter Keep Ratio (%)")
    plt.title("Task 1: Deep Specialization\n(e.g., downstream fine-tuning task)")
    plt.xticks(layers)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot Task 2 Layer-wise Keep Ratios
    plt.subplot(1, 2, 2)
    plt.bar(layers - 0.2, [r * 100 for r in results["task2"]["GQ"]["layer_keep_ratios"]], width=0.4, label="GQ (Ours)", color="royalblue")
    plt.bar(layers + 0.2, [k_target * 100] * num_layers, width=0.4, label="LQ (Rigid)", color="darkorange", alpha=0.8)
    plt.xlabel("Transformer Layer Index")
    plt.ylabel("Parameter Keep Ratio (%)")
    plt.title("Task 2: Middle Specialization\n(e.g., structural semantic adjustments)")
    plt.xticks(layers)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.suptitle(f"Sparsity Allocation under GQ (Ours) vs. LQ (Rigid) at k = {k_target:.2f} in a 12-layer Transformer", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("./results/nlp_simulation_sparsity.png", dpi=300)
    print("Simulation complete! Results saved to ./results/nlp_simulation_metrics.json and ./results/nlp_simulation_sparsity.png")

if __name__ == "__main__":
    main()
