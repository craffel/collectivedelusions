import json
import os

ranks = [4, 8, 16]
modes = ["so_lora", "so_lora_sam"]
coeffs = ["coeffs_0.5_0.5", "coeffs_0.7_0.7", "coeffs_1.0_1.0"]

print("=== SO-LoRA Rank Sweep Results ===")
print("| Method | Rank | Coeffs | Task 1 Acc | Task 2 Acc | Multi-Task Avg |")
print("|--------|------|--------|------------|------------|----------------|")

for mode in modes:
    for r in ranks:
        suffix = f"_r{r}" if r != 8 else ""
        filename = f"results_{mode}{suffix}.json"
        
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            for c in coeffs:
                if c in data:
                    res = data[c]
                    t1 = res["task1_acc"]
                    t2 = res["task2_acc"]
                    avg = res["avg_acc"]
                    coeff_str = c.replace("coeffs_", "")
                    print(f"| {mode.upper()} | r={r} | {coeff_str} | {t1:.2f}% | {t2:.2f}% | {avg:.2f}% |")
        else:
            print(f"| {mode.upper()} | r={r} | (not finished yet) |")
