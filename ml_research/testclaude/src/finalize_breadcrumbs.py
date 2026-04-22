"""Fill in Breadcrumbs placeholders in paper.tex using the best config from results/breadcrumbs.json."""
import json
import re
import sys

RESULTS = "/fsx/craffel/collectivedelusions/ml_research/testclaude/results/breadcrumbs.json"
PAPER = "/fsx/craffel/collectivedelusions/ml_research/testclaude/paper/paper.tex"

with open(RESULTS) as f:
    data = json.load(f)

best = data["best"]
per = best["per_task"]
avg = best["avg"]
print(f"Best: ds={best['drop_small']} dl={best['drop_large']} a={best['alpha']} avg={avg*100:.2f}")
print("Per-task:", {t: f"{v*100:.1f}" for t, v in per.items()})

# Table columns ordered: MNIST CIFAR10 CIFAR100 SVHN FashionMNIST EuroSAT GTSRB DTD
mapping = [
    ("BCMNIST", per["MNIST"]),
    ("BCC100", per["CIFAR100"]),      # replace longer before shorter
    ("BCC10", per["CIFAR10"]),
    ("BCSVHN", per["SVHN"]),
    ("BCFMN", per["FashionMNIST"]),
    ("BCEURO", per["EuroSAT"]),
    ("BCGTS", per["GTSRB"]),
    ("BCDTD", per["DTD"]),
    ("BCAVG", avg),
]

with open(PAPER) as f:
    tex = f.read()

for token, val in mapping:
    val_str = f"{val*100:.1f}"
    if token not in tex:
        print(f"WARN: token {token} not found in paper.tex")
        continue
    tex = tex.replace(token, val_str)
    print(f"  {token} -> {val_str}")

with open(PAPER, "w") as f:
    f.write(tex)
print("Paper updated.")
