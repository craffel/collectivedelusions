import os
import json
import numpy as np
from merge_and_evaluate_multidimensional import run_multidim_experiment

seeds = [42, 43, 44, 45, 46]
metrics = ["cka", "mse", "cosine", "mmd", "oracle", "wa"]

all_seed_results = []

for seed in seeds:
    print(f"\n==========================================")
    print(f"RUNNING SWEEP FOR SEED {seed}")
    print(f"==========================================\n")
    res = run_multidim_experiment(cal_size=128, target_layer="layer4", seed=seed)
    all_seed_results.append(res)
    
# Aggregate and print results
print("\n==========================================")
print("AGGREGATED MULTIDIMENSIONAL RESULTS (5 SEEDS)")
print("==========================================\n")

aggregated = {m: [] for m in metrics}

for res in all_seed_results:
    full_res = res["full_test_results"]
    for m in ["cka", "mse", "cosine", "mmd"]:
        aggregated[m].append(full_res[m]["avg_acc"])
    aggregated["oracle"].append(full_res["oracle"]["avg_acc"])
    aggregated["wa"].append(full_res["wa"]["avg_acc"])

summary = {}
for m in metrics:
    vals = np.array(aggregated[m])
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    summary[m] = {"mean": mean, "std": std, "values": vals.tolist()}
    print(f"M-AOS {m.upper() if m != 'oracle' and m != 'wa' else m.capitalize()}: {mean:.2f}% ± {std:.2f}%")

# Save summary to json
with open("results_multidim_summary.json", "w") as f:
    json.dump({"summary": summary, "seeds_data": all_seed_results}, f, indent=4)
print("\nSaved aggregated multidimensional results to results_multidim_summary.json")
