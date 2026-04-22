"""Inject DELLA best config into results/results.json for plot reuse."""
import json, os
REPO = "/fsx/craffel/collectivedelusions/ml_research/testclaude"
della = json.load(open(os.path.join(REPO, "results", "della.json")))
results = json.load(open(os.path.join(REPO, "results", "results.json")))
best = della["best"]
results["methods"]["della_best"] = {
    "avg": best["avg"],
    "per_task": best["per_task"],
    "config": {"p_low": best["p_low"], "p_high": best["p_high"], "alpha": best["alpha"]},
}
with open(os.path.join(REPO, "results", "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Injected della_best avg={best['avg']*100:.2f}")
