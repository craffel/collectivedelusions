import numpy as np
from run_experiments import run_simulation_seed

raw_results = {m: {"accuracy": [], "jitter": []} for m in ["Expert Ceiling", "Uniform Merging", "SABLE", "ChemMerge", "Momentum-Merge"]}

print("Running 10 seeds uncalibrated...")
for seed in range(10):
    res = run_simulation_seed(seed=seed)
    for m in raw_results.keys():
        raw_results[m]["accuracy"].append(res[m]["accuracy"])
        raw_results[m]["jitter"].append(res[m]["jitter"])

print("\n" + "="*80)
print(f"{'Method':<20} | {'Accuracy Mean (%)':<15} | {'Accuracy Std (%)':<15} | {'Jitter Mean (MSE)':<15}")
print("="*80)
for m in raw_results.keys():
    accs = np.array(raw_results[m]["accuracy"])
    jits = np.array(raw_results[m]["jitter"])
    print(f"{m:<20} | {np.mean(accs)*100:<15.2f} | {np.std(accs)*100:<15.2f} | {np.mean(jits):<15.6f}")
print("="*80)
