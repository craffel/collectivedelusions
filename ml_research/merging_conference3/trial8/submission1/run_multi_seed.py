import torch
import numpy as np
import sys
from run_experiments import run_regime

subspaces_orth = {
    0: (0, 48),
    1: (48, 96),
    2: (96, 144),
    3: (144, 192)
}

subspaces_overlap = {
    0: (0, 96),
    1: (32, 128),
    2: (64, 160),
    3: (96, 192)
}

seeds = [42, 43, 44]

# We will collect metrics for both orth and overlap
methods = [
    'ceil_homo', 'uni_homo', 'pfsr_homo', 'pfsr_hetero', 'pfsr_mbh_hetero',
    'sable_e_homo', 'sable_l_homo', 'sps_homo', 'hm_homo'
]

print("Starting multi-seed evaluation...")
sys.stdout.flush()

orth_results = {m: [] for m in methods}
overlap_results = {m: [] for m in methods}

# Also gather HyperMerge tuned for overlap
overlap_hm_tuned = []

for seed in seeds:
    print(f"Evaluating Seed {seed}...")
    sys.stdout.flush()
    
    # Run Orthogonal
    res_orth = run_regime(subspaces_orth, c_hyperbolic=0.1, seed=seed)
    for m in methods:
        orth_results[m].append(res_orth[m] * 100)
        
    # Run Overlapping default
    res_overlap = run_regime(subspaces_overlap, c_hyperbolic=0.1, seed=seed)
    for m in methods:
        overlap_results[m].append(res_overlap[m] * 100)
        
    # Run Overlapping tuned (c=0.2, tau=0.08)
    res_overlap_tuned = run_regime(subspaces_overlap, c_hyperbolic=0.2, tau=0.08, seed=seed)
    overlap_hm_tuned.append(res_overlap_tuned['hm_homo'] * 100)

print("\n--- STATISTICAL RESULTS (Mean ± Std) ---")

def print_stats(name, values):
    mean = np.mean(values)
    std = np.std(values)
    print(f"{name:<25}: {mean:.2f}% ± {std:.2f}%")

print("\n[A. ORTHOGONAL SUB-SANDBOX]")
print_stats("Expert Ceiling", orth_results['ceil_homo'])
print_stats("Uniform Merging (Static)", orth_results['uni_homo'])
print_stats("PFSR (Homo)", orth_results['pfsr_homo'])
print_stats("PFSR (Hetero)", orth_results['pfsr_hetero'])
print_stats("PFSR+MBH (Hetero)", orth_results['pfsr_mbh_hetero'])
print_stats("SABLE Early (Homo)", orth_results['sable_e_homo'])
print_stats("SABLE Late (Homo)", orth_results['sable_l_homo'])
print_stats("SPS-ZCA (SOTA Euclidean)", orth_results['sps_homo'])
print_stats("HyperMerge (Ours)", orth_results['hm_homo'])

print("\n[B. OVERLAPPING SUB-SANDBOX]")
print_stats("Expert Ceiling", overlap_results['ceil_homo'])
print_stats("Uniform Merging (Static)", overlap_results['uni_homo'])
print_stats("PFSR (Homo)", overlap_results['pfsr_homo'])
print_stats("PFSR (Hetero)", overlap_results['pfsr_hetero'])
print_stats("PFSR+MBH (Hetero)", overlap_results['pfsr_mbh_hetero'])
print_stats("SABLE Early (Homo)", overlap_results['sable_e_homo'])
print_stats("SABLE Late (Homo)", overlap_results['sable_l_homo'])
print_stats("SPS-ZCA (SOTA Euclidean)", overlap_results['sps_homo'])
print_stats("HyperMerge (Ours, c=0.1)", overlap_results['hm_homo'])
print_stats("HyperMerge (Ours, Tuned)", overlap_hm_tuned)
sys.stdout.flush()
