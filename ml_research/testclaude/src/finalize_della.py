"""Replace DELLA placeholder tokens in paper/paper.tex with best-config results."""
import json, os, sys

REPO = "/fsx/craffel/collectivedelusions/ml_research/testclaude"
results_path = os.path.join(REPO, "results", "della.json")
tex_path = os.path.join(REPO, "paper", "paper.tex")

with open(results_path) as f:
    data = json.load(f)
best = data["best"]
per_task = best["per_task"]
avg = best["avg"]
cfg = f"(p_low={best['p_low']}, p_high={best['p_high']}, alpha={best['alpha']})"
print(f"Best DELLA: {cfg} avg={avg*100:.2f}")
for k, v in per_task.items():
    print(f"  {k}: {v*100:.2f}")

mapping = {
    "DELLAMNIST": f"{per_task['MNIST']*100:.1f}",
    "DELLAC100":  f"{per_task['CIFAR100']*100:.1f}",
    "DELLAC10":   f"{per_task['CIFAR10']*100:.1f}",
    "DELLASVHN":  f"{per_task['SVHN']*100:.1f}",
    "DELLAFMN":   f"{per_task['FashionMNIST']*100:.1f}",
    "DELLAEURO":  f"{per_task['EuroSAT']*100:.1f}",
    "DELLAGTS":   f"{per_task['GTSRB']*100:.1f}",
    "DELLADTD":   f"{per_task['DTD']*100:.1f}",
    "DELLAAVG":   f"{avg*100:.1f}",
}

with open(tex_path) as f:
    tex = f.read()

# Replace longer tokens first to avoid prefix clash (DELLAC100 before DELLAC10).
for tok in sorted(mapping.keys(), key=lambda s: -len(s)):
    if tok in tex:
        tex = tex.replace(tok, mapping[tok])
        print(f"  replaced {tok} -> {mapping[tok]}")
    else:
        print(f"  WARN: {tok} not found in paper.tex")

with open(tex_path, "w") as f:
    f.write(tex)

print(f"\nPaper updated: {tex_path}")
