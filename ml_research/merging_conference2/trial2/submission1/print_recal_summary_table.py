import json
import os

if not os.path.exists("results_summary.json"):
    print("Error: results_summary.json not found.")
    exit(1)
if not os.path.exists("results_recal_summary.json"):
    print("Error: results_recal_summary.json not found.")
    exit(1)

with open("results_summary.json", "r") as f:
    orig_summary = json.load(f)
with open("results_recal_summary.json", "r") as f:
    recal_summary = json.load(f)

sizes = [8, 16, 32, 64, 128, 256]
layer = "layer4"

print(f"=========================================================================================")
print(f"      COMPARATIVE BATCH NORMALIZATION PERFORMANCE ON {layer.upper()}")
print(f"=========================================================================================")
print(f"| {'Size (N)':<8} | {'Orig CKA Acc':<15} | {'Recal CKA Acc':<15} | {'Orig Oracle':<12} | {'Recal Oracle':<12} | {'Gain (CKA)':<10} |")
print(f"|{'-'*10}|{'-'*17}|{'-'*17}|{'-'*14}|{'-'*14}|{'-'*12}|")

for size in sizes:
    config_name = f"size{size}_layer{layer}"
    
    orig_d = orig_summary.get(config_name, {}).get("cka", {})
    orig_o = orig_summary.get(config_name, {}).get("oracle", {})
    
    recal_d = recal_summary.get(config_name, {}).get("cka", {})
    recal_o = recal_summary.get(config_name, {}).get("oracle", {})
    
    orig_cka = orig_d.get("mean_acc", 0.0)
    orig_cka_std = orig_d.get("std_acc", 0.0)
    orig_ora = orig_o.get("mean_acc", 0.0)
    
    recal_cka = recal_d.get("mean_acc", 0.0)
    recal_cka_std = recal_d.get("std_acc", 0.0)
    recal_ora = recal_o.get("mean_acc", 0.0)
    
    gain = recal_cka - orig_cka
    
    orig_cka_str = f"{orig_cka:.2f}% ± {orig_cka_std:.2f}%" if orig_cka else "N/A"
    recal_cka_str = f"{recal_cka:.2f}% ± {recal_cka_std:.2f}%" if recal_cka else "N/A"
    orig_ora_str = f"{orig_ora:.2f}%" if orig_ora else "N/A"
    recal_ora_str = f"{recal_ora:.2f}%" if recal_ora else "N/A"
    gain_str = f"{gain:+.2f}%" if orig_cka and recal_cka else "N/A"
    
    print(f"| N={size:<5} | {orig_cka_str:<15} | {recal_cka_str:<15} | {orig_ora_str:<12} | {recal_ora_str:<12} | {gain_str:<10} |")
print(f"=========================================================================================")
