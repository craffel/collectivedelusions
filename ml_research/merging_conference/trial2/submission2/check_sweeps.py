import json
import os

files = ["results_so_lora_sam_r4.json", "results_so_lora_sam_r16.json"]

print("=== SWEEP RESULTS ===")
for f in files:
    if os.path.exists(f):
        print(f"\n[{f}]")
        with open(f, "r") as fh:
            data = json.load(fh)
            print(json.dumps(data, indent=4))
    else:
        print(f"\n[{f}] Not found yet.")
