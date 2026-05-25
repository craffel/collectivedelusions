import json
import os

modes = ["standard", "sam", "so_lora", "so_lora_sam"]
coeffs_keys = ["coeffs_0.5_0.5", "coeffs_0.7_0.7", "coeffs_1.0_1.0"]

print("=== ACCURACY RESULTS COMPILED ===")
print("| Mode | Coeffs | Task 1 Acc | Task 2 Acc | Multi-Task Avg |")
print("|------|--------|------------|------------|----------------|")

latex_rows = []

for mode in modes:
    filename = f"results_{mode}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        for key in coeffs_keys:
            if key in data:
                res = data[key]
                t1 = res["task1_acc"]
                t2 = res["task2_acc"]
                avg = res["avg_acc"]
                coeff_str = key.replace("coeffs_", "")
                print(f"| {mode.upper()} | {coeff_str} | {t1:.2f}% | {t2:.2f}% | {avg:.2f}% |")
                latex_rows.append(f"{mode.upper()} & {coeff_str} & {t1:.2f}\\% & {t2:.2f}\\% & {avg:.2f}\\% \\\\")
    else:
        print(f"| {mode.upper()} | (not finished yet) |")

print("\n=== LATEX TABLE ROWS ===")
for r in latex_rows:
    print(r)
