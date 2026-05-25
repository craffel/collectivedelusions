import json
import glob

files = glob.glob("results_rho_*.json")

print(f"{'Rho':<6} {'LR':<6} | {'Clean Accs':<35} {'Avg Clean':<10} | {'Corr Accs':<35} {'Avg Corr':<10}")
print("-" * 115)

results = []

for f in sorted(files):
    # Parse rho and lr from file name
    parts = f.replace(".json", "").split("_")
    rho = float(parts[2])
    lr = float(parts[4])
    
    with open(f, "r") as jf:
        data = json.load(jf)
        
    u_sasla = data["u_sasla"]
    clean = u_sasla["clean"]
    corr = u_sasla["corrupted"]
    
    clean_vals = [clean[t] for t in ["mnist", "fashion", "cifar10"]]
    corr_vals = [corr[t] for t in ["mnist", "fashion", "cifar10"]]
    
    avg_clean = sum(clean_vals) / len(clean_vals)
    avg_corr = sum(corr_vals) / len(corr_vals)
    
    results.append({
        "rho": rho,
        "lr": lr,
        "clean_vals": clean_vals,
        "avg_clean": avg_clean,
        "corr_vals": corr_vals,
        "avg_corr": avg_corr
    })

# Sort by avg_corr descending
results.sort(key=lambda x: x["avg_corr"], reverse=True)

for r in results:
    clean_str = ", ".join([f"{v:.4f}" for v in r["clean_vals"]])
    corr_str = ", ".join([f"{v:.4f}" for v in r["corr_vals"]])
    print(f"{r['rho']:<6.2f} {r['lr']:<6.4f} | [{clean_str}] {r['avg_clean']:<10.4f} | [{corr_str}] {r['avg_corr']:<10.4f}")
