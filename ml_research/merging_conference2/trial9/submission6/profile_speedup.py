import torch
import numpy as np
import time

def profile_sorting_speedup():
    print("======================================================================")
    print("Empirical Speedup Profiling: Standard WCPR vs. QR-SC-WCPR Sorting Phase")
    print("======================================================================")
    
    # Simulate realistic ResNet-18 layer dimensions
    # Convolutional filter size: channels_out * channels_in * kernel_height * kernel_width
    # E.g., for channels_out = 512, channels_in = 512, kernel = 3x3:
    # M = 3 * 3 * 512 = 4608 elements per filter channel
    M = 4608
    num_runs = 50
    
    # Sweep active ratio (p_c)
    p_c_values = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    
    print(f"Channel dimension (M): {M}")
    print(f"Number of profiling runs per setting: {num_runs}\n")
    
    results = []
    
    for p_c in p_c_values:
        num_active = int(M * p_c)
        if num_active == 0:
            num_active = 1
            
        # 1. Standard WCPR Sorting: Sorts the entire flat channel (M elements)
        t_start_wcpr = time.perf_counter()
        for _ in range(num_runs):
            # Create dummy flat channel of size M
            mc = torch.randn(M)
            # Sort entire tensor
            Ic = torch.argsort(mc)
        t_end_wcpr = time.perf_counter()
        time_wcpr = (t_end_wcpr - t_start_wcpr) / num_runs * 1000.0 # in milliseconds
        
        # 2. QR-SC-WCPR Sorting: Only sorts the active subset of size p_c * M
        t_start_qr_sc = time.perf_counter()
        for _ in range(num_runs):
            # Create active parameters of size num_active
            mc_active = torch.randn(num_active)
            # Sort only the active subset
            Ic_active = torch.argsort(mc_active)
        t_end_qr_sc = time.perf_counter()
        time_qr_sc = (t_end_qr_sc - t_start_qr_sc) / num_runs * 1000.0 # in milliseconds
        
        speedup = time_wcpr / (time_qr_sc + 1e-12)
        sparsity = (1.0 - p_c) * 100.0
        
        # Compute theoretical speedup according to our derivation:
        # speedup_theory = log2(M) / (p_c * (log2(M) + log2(p_c)))
        log_M = np.log2(M)
        log_pc = np.log2(p_c)
        denom = p_c * (log_M + log_pc)
        speedup_theory = log_M / denom if denom > 0 else float('inf')
        
        results.append({
            "p_c": p_c,
            "sparsity": sparsity,
            "num_active": num_active,
            "time_wcpr_ms": time_wcpr,
            "time_qr_sc_ms": time_qr_sc,
            "speedup_empirical": speedup,
            "speedup_theoretical": speedup_theory
        })
        
        print(f"Active Ratio (p_c): {p_c:.2f} ({sparsity:.1f}% Sparsity) | Active elements: {num_active}")
        print(f"  Standard WCPR Sort Time: {time_wcpr:.4f} ms")
        print(f"  QR-SC-WCPR Sort Time:   {time_qr_sc:.4f} ms")
        print(f"  Empirical Speedup:      {speedup:.2f}x (Theoretical: {speedup_theory:.2f}x)\n")

    # Generate a beautiful Markdown table
    md_table = []
    md_table.append("# Empirical Sorting Latency and Speedup Analysis")
    md_table.append("This document records the empirical benchmarking of the sorting-phase computational complexity for Standard WCPR versus our proposed QR-SC-WCPR algorithm. Measurements were conducted on the local CPU node (4 CPUs, 16 GB RAM).")
    md_table.append("")
    md_table.append(f"### Benchmarking Configuration:")
    md_table.append(f"- **Channel Dimension (M):** {M} (corresponding to a typical ResNet-18 filter channel size)")
    md_table.append(f"- **Repetitions:** {num_runs} independent trials per active ratio")
    md_table.append("")
    md_table.append("| Active Ratio ($p_c$) | Sparsity (%) | Active Parameters ($N_a$) | Standard WCPR (ms) | QR-SC-WCPR (ms) | Empirical Speedup | Theoretical Speedup |")
    md_table.append("|:--------------------:|:------------:|:------------------------:|:------------------:|:---------------:|:-----------------:|:------------------:|")
    
    for r in results:
        md_table.append(
            f"| {r['p_c']:.2f} | {r['sparsity']:.1f}% | {r['num_active']} | {r['time_wcpr_ms']:.4f} | {r['time_qr_sc_ms']:.4f} | {r['speedup_empirical']:.2f}x | {r['speedup_theoretical']:.2f}x |"
        )
        
    md_content = "\n".join(md_table)
    with open("results/profiling_results.md", "w") as f:
        f.write(md_content)
    print("Saved profiling results to results/profiling_results.md")

if __name__ == "__main__":
    profile_sorting_speedup()
