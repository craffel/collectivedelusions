import json
import numpy as np

with open('flatq_merge_results.json', 'r') as f:
    data = json.load(f)

# Let's inspect seed 42, rho 0.05 (optimal), bit 4 (and bit 8) for FlatQ-Merge
for bit in ["8", "4"]:
    for rho in ["0.0", "0.05", "0.2"]:
        try:
            lambdas = np.array(data["results"]["42"][rho][bit]["FlatQ-Merge"]["lambdas"])
            # lambdas shape: (L, K) where L is number of layers (14), K is number of tasks (4)
            sums = np.sum(lambdas, axis=1)
            print(f"=== Bit-width {bit}, Rho {rho} ===")
            print(f"Number of layers: {len(sums)}")
            print(f"Layer-wise sums: {sums}")
            print(f"Mean sum: {np.mean(sums):.4f}, Std sum: {np.std(sums):.4f}")
            print(f"Min sum: {np.min(sums):.4f} (layer {np.argmin(sums)}), Max sum: {np.max(sums):.4f} (layer {np.argmax(sums)})")
            print(f"Individual lambdas range: [{np.min(lambdas):.4f}, {np.max(lambdas):.4f}]")
            print(f"Count near 0.0 (<0.01): {np.sum(lambdas < 0.01)}")
            print(f"Count near 1.0 (>0.99): {np.sum(lambdas > 0.99)}")
            print("-" * 50)
        except Exception as e:
            print(f"Config {bit} bit, rho {rho} failed to analyze: {e}")
