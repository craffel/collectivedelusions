import json

with open("results_summary.json", "r") as f:
    summary = json.load(f)

print(f"{'Config':<25} | {'CKA Acc':<12} | {'MSE Acc':<12} | {'Cosine Acc':<12} | {'MMD Acc':<12} | {'Oracle Acc':<12} | {'WA Acc':<12}")
print("-" * 110)

for config, d in sorted(summary.items()):
    # e.g., size64_layerlayer4
    parts = config.split("_")
    size = parts[0].replace("size", "")
    layer = parts[1].replace("layer", "")
    config_name = f"N={size}, L={layer}"
    
    cka = f"{d['cka']['mean_acc']:.2f}% ± {d['cka']['std_acc']:.2f}%"
    mse = f"{d['mse']['mean_acc']:.2f}% ± {d['mse']['std_acc']:.2f}%"
    cosine = f"{d['cosine']['mean_acc']:.2f}% ± {d['cosine']['std_acc']:.2f}%"
    mmd = f"{d['mmd']['mean_acc']:.2f}% ± {d['mmd']['std_acc']:.2f}%"
    oracle = f"{d['oracle']['mean_acc']:.2f}% ± {d['oracle']['std_acc']:.2f}%"
    wa = f"{d['wa']['mean_acc']:.2f}%"
    
    print(f"{config_name:<25} | {cka:<12} | {mse:<12} | {cosine:<12} | {mmd:<12} | {oracle:<12} | {wa:<12}")
