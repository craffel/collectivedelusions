import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from experiment import MultiTaskResNet18, collect_stats, register_inference_hooks

def count_parameters(mode, task_name, scaling_factors, threshold=1.30):
    # Counts the number of calibration parameters stored per task
    num_bn_layers = 0
    total_channels = 0
    calibrated_layers = 0
    calibrated_channels = 0
    
    task_data = scaling_factors[task_name]
    for layer_name, stats in task_data.items():
        num_bn_layers += 1
        # In ResNet-18, we can infer channel size from the stats if we have them,
        # but since we want the exact channel sizes of each layer:
        # We can map layer names to channel sizes
        # bn1 is 64, layer1.*.bn* is 64, layer2.*.bn* is 128, layer3.*.bn* is 256, layer4.*.bn* is 512
        if "bn1" in layer_name and "layer" not in layer_name:
            channels = 64
        elif "layer1" in layer_name:
            channels = 64
        elif "layer2" in layer_name:
            channels = 128
        elif "layer3" in layer_name:
            channels = 256
        elif "layer4" in layer_name:
            channels = 512
        else:
            channels = 64
            
        total_channels += channels
        
        # Check if calibrated under TSC
        gamma = stats.get("gamma", 1.0)
        if gamma >= threshold:
            calibrated_layers += 1
            calibrated_channels += channels

    if mode == "none":
        return 0
    elif mode == "tcac":
        # Stores channel-wise mu_orig, sigma_orig, mu_merged, sigma_merged (4 * C floats)
        # Even if mathematically optimized to a scale and a shift, it's 2 * C floats
        # Let's report the standard unoptimized formulation (4 * C) and optimized (2 * C)
        # We will report the optimized (2 * C) to be conservative and fair
        return 2 * total_channels
    elif mode == "sac":
        # Stores channel-wise scale (1 * C floats)
        return total_channels
    elif mode == "lsc":
        # Stores exactly 1 scalar float per layer
        return num_bn_layers
    elif mode == "tsc":
        # Stores 1 scalar float per calibrated layer (no shift, only scale)
        # Plus 1 bit per layer for the boolean mask (which is negligible, < 1 float)
        return calibrated_layers
    return 0

def benchmark_inference(model, task_calibration, mode, device, num_runs=500, warmups=100):
    # Register inference hooks
    handles = register_inference_hooks(model, task_calibration, mode)
    
    # Generate dummy input batch (size 128)
    dummy_input = torch.randn(128, 3, 32, 32).to(device)
    
    model.eval()
    with torch.no_grad():
        # Warmup runs
        for _ in range(warmups):
            _ = model(dummy_input, "mnist")
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        # Benchmark runs
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input, "mnist")
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
    # Remove hooks
    for handle in handles:
        handle.remove()
        
    total_time_ms = (end_time - start_time) * 1000.0
    avg_latency_ms = total_time_ms / num_runs
    return avg_latency_ms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on device: {device}")
    
    # Load scaling factors
    if not os.path.exists("scaling_factors.json"):
        print("Error: scaling_factors.json not found! Run evaluation first.")
        return
        
    with open("scaling_factors.json", "r") as f:
        scaling_factors = json.load(f)
        
    # Instantiate model
    model = MultiTaskResNet18(["mnist", "fashion", "cifar10"]).to(device)
    
    # Standardize task_calibration format from scaling_factors
    # The stats collection expects keys like mu_orig, sigma_orig, mu_merged, sigma_merged
    # Let's generate realistic stats dictionary for the benchmark
    task_calibration = {}
    for layer_name in scaling_factors["mnist"].keys():
        # Infer channel size
        if "bn1" in layer_name and "layer" not in layer_name:
            C = 64
        elif "layer1" in layer_name:
            C = 64
        elif "layer2" in layer_name:
            C = 128
        elif "layer3" in layer_name:
            C = 256
        elif "layer4" in layer_name:
            C = 512
        else:
            C = 64
            
        task_calibration[layer_name] = {
            'mu_orig': torch.zeros(1, C, 1, 1).to(device),
            'sigma_orig': torch.ones(1, C, 1, 1).to(device),
            'mu_merged': torch.zeros(1, C, 1, 1).to(device),
            'sigma_merged': torch.ones(1, C, 1, 1).to(device),
            'sigma_orig_scalar': torch.tensor(1.0).to(device),
            'sigma_merged_scalar': torch.tensor(1.0).to(device)
        }
        
    # For TSC, we only calibrate layers where gamma >= 1.30
    # Let's filter task_calibration to create task_calibration_tsc
    task_calibration_tsc = {}
    for layer_name, stats in scaling_factors["mnist"].items():
        if stats.get("gamma", 1.0) >= 1.30:
            task_calibration_tsc[layer_name] = task_calibration[layer_name]
            
    # Measure Latencies
    print("\n--- Running Latency Benchmarks ---")
    latency_none = benchmark_inference(model, task_calibration, 'none', device)
    print(f"NONE Latency: {latency_none:.4f} ms per batch")
    
    latency_tcac = benchmark_inference(model, task_calibration, 'tcac', device)
    print(f"TCAC Latency: {latency_tcac:.4f} ms per batch")
    
    latency_sac = benchmark_inference(model, task_calibration, 'sac', device)
    print(f"SAC Latency: {latency_sac:.4f} ms per batch")
    
    latency_lsc = benchmark_inference(model, task_calibration, 'lsc', device)
    print(f"LSC Latency: {latency_lsc:.4f} ms per batch")
    
    # For TSC, we use the registered hooks ONLY for the subset of layers
    # We can use the 'lsc' mode but only registered on the subset of layers in task_calibration_tsc
    handles = []
    # Let's measure TSC latency
    # We can register the hook manually or by using the function
    latency_tsc = benchmark_inference(model, task_calibration_tsc, 'lsc', device)
    print(f"TSC Latency: {latency_tsc:.4f} ms per batch")
    
    # Calculate parameter storage sizes
    print("\n--- Calculating Parameter Storage ---")
    params_none = count_parameters("none", "mnist", scaling_factors)
    params_tcac = count_parameters("tcac", "mnist", scaling_factors)
    params_sac = count_parameters("sac", "mnist", scaling_factors)
    params_lsc = count_parameters("lsc", "mnist", scaling_factors)
    params_tsc = count_parameters("tsc", "mnist", scaling_factors, threshold=1.30)
    
    print(f"NONE: {params_none} parameters")
    print(f"TCAC (optimized scale+shift): {params_tcac} parameters")
    print(f"SAC: {params_sac} parameters")
    print(f"LSC (Ours): {params_lsc} parameters")
    print(f"TSC (Ours, threshold=1.30): {params_tsc} parameters")
    
    # Measure Calibration Time
    print("\n--- Measuring Calibration Time ---")
    dummy_batch = torch.randn(128, 3, 32, 32).to(device)
    
    # Warmup
    _ = collect_stats(model, dummy_batch, "mnist", device=device)
    
    start_cal = time.perf_counter()
    for _ in range(10):
        _ = collect_stats(model, dummy_batch, "mnist", device=device)
    end_cal = time.perf_counter()
    avg_cal_time_ms = ((end_cal - start_cal) / 10.0) * 1000.0
    print(f"Average Calibration Time (128 samples): {avg_cal_time_ms:.2f} ms")
    
    # Report summary
    results = {
        "device": device,
        "latency": {
            "none": latency_none,
            "tcac": latency_tcac,
            "sac": latency_sac,
            "lsc": latency_lsc,
            "tsc": latency_tsc
        },
        "storage": {
            "none": params_none,
            "tcac": params_tcac,
            "sac": params_sac,
            "lsc": params_lsc,
            "tsc": params_tsc
        },
        "calibration_time_ms": avg_cal_time_ms
    }
    
    with open("efficiency_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = ["NONE", "TCAC", "SAC", "LSC", "TSC"]
    latencies = [latency_none, latency_tcac, latency_sac, latency_lsc, latency_tsc]
    colors = ["#7f7f7f", "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    
    # Plot Latency Comparison
    bars1 = ax1.bar(methods, latencies, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_ylabel("Inference Latency per Batch (ms)", fontsize=12)
    ax1.set_title("Inference Latency Comparison\n(Batch Size = 128, ResNet-18)", fontsize=13, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.3f} ms", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # Plot Storage Comparison (Log scale)
    storages = [params_none, params_tcac, params_sac, params_lsc, params_tsc]
    # For none, store 0 but set to 1 for log scale plotting
    plot_storages = [max(1, s) for s in storages]
    
    bars2 = ax2.bar(methods, plot_storages, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel("Calibration Parameter Storage per Task (Log Scale)", fontsize=12)
    ax2.set_title("Calibration Storage Complexity per Task\n(Number of Parameters, Log Scale)", fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, bar in enumerate(bars2):
        yval = storages[i]
        ax2.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() * 1.1 if yval > 0 else 1.2, f"{yval}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("efficiency_comparison.png", dpi=300)
    print("\nBenchmark completed successfully! Saved efficiency_results.json and efficiency_comparison.png.")

if __name__ == "__main__":
    main()
