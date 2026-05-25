import json
import os

modes = ["standard", "sam", "so_lora", "so_lora_sam"]
coeffs_keys = ["0.5", "0.7", "1.0"]

# We will load results from the JSONs
results = {}
for mode in modes:
    filename = f"results_{mode}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        results[mode] = data
    else:
        results[mode] = {}

# Read submission.tex
if os.path.exists("submission.tex"):
    with open("submission.tex", "r") as f:
        tex = f.read()
else:
    print("submission.tex not found!")
    exit(1)

# Mapping placeholders
# Placeholders are of the form [SAM_0.5_T1], [SO_0.5_T1], etc.
# Note the prefix matches:
# standard -> STANDARD
# sam -> SAM
# so_lora -> SO
# so_lora_sam -> SOSAM

prefixes = {
    "standard": "STANDARD",
    "sam": "SAM",
    "so_lora": "SO",
    "so_lora_sam": "SOSAM"
}

for mode in modes:
    prefix = prefixes[mode]
    mode_data = results[mode]
    for c in coeffs_keys:
        key = f"coeffs_{c}_{c}"
        if key in mode_data:
            t1 = f"{mode_data[key]['task1_acc']:.2f}"
            t2 = f"{mode_data[key]['task2_acc']:.2f}"
            avg = f"{mode_data[key]['avg_acc']:.2f}"
            
            # Replace placeholders
            tex = tex.replace(f"[{prefix}_{c}_T1]", t1)
            tex = tex.replace(f"[{prefix}_{c}_T2]", t2)
            tex = tex.replace(f"[{prefix}_{c}_AVG]", avg)
        else:
            print(f"Warning: {key} not found for {mode}")

with open("submission.tex", "w") as f:
    f.write(tex)

print("Placeholders updated in submission.tex!")
