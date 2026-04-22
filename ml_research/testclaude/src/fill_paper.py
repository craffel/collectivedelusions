"""Substitute \\RESULT{name} placeholders in paper.tex with actual numbers."""
import json, os, re

ROOT = "/fsx/craffel/collectivedelusions/ml_research/testclaude"
PAPER = os.path.join(ROOT, "paper/paper.tex")

with open(os.path.join(ROOT, "results/extras_baselines.json")) as f:
    base = json.load(f)
with open(os.path.join(ROOT, "results/extras_taskcount.json")) as f:
    tc = json.load(f)
with open(os.path.join(ROOT, "results/taskcount_k.json")) as f:
    tck = json.load(f)


def pct(x):
    return f"{x * 100:.1f}"


vals = {}
cols = ["MNIST", "CIFAR10", "CIFAR100", "SVHN", "FashionMNIST", "EuroSAT", "GTSRB", "DTD"]
keys = ["mnist", "cifar10", "cifar100", "svhn", "fm", "eurosat", "gtsrb", "dtd"]

for prefix, method_key in [("dare", "dare_best"), ("dt", "dare_ties_best"), ("fi", "fisher_approx")]:
    pt = base["methods"][method_key]["per_task"]
    for c, k in zip(cols, keys):
        vals[f"{prefix}_{k}"] = pct(pt[c])
    vals[f"{prefix}_avg"] = pct(base["methods"][method_key]["avg"])

for T in [2, 4, 6]:
    for meth, prefix in [("task_arith", "ta"), ("ties", "ti"), ("actmat", "ac"), ("scale", "sc")]:
        if prefix == "sc":
            vals[f"{prefix}_t{T}"] = pct(tck[f"T{T}_best"]["avg"])
        else:
            vals[f"{prefix}_t{T}"] = pct(tc["task_count"][f"{meth}_T{T}"]["avg"])
    vals[f"ks_t{T}"] = f"{tck[f'T{T}_best']['k']:.1f}"

for k_str in ["1.0", "0.5", "0.3", "0.2", "0.1", "0.05"]:
    vals[f"sa_k{k_str}"] = f"{base['sign_agreement'][f'k{k_str}']:.2f}"

sc_t2 = tck["T2_best"]["avg"] * 100
ac_t2 = tc["task_count"]["actmat_T2"]["avg"] * 100
vals["scale_minus_actmat_T2"] = f"{abs(sc_t2 - ac_t2):.1f}"

print("Substitutions:")
for k in sorted(vals):
    print(f"  {k:30s} -> {vals[k]}")

with open(PAPER) as f:
    tex = f.read()

for key, val in vals.items():
    tex = tex.replace("\\RESULT{" + key + "}", val)

remaining = sorted(set(re.findall(r"\\RESULT\{([^}]+)\}", tex)))
if remaining:
    print(f"\nUNREPLACED: {remaining}")

with open(PAPER, "w") as f:
    f.write(tex)

print("\nDone.")
