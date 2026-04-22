"""Replace T=6 random-subset placeholder tokens in paper/paper.tex."""
import json, os

REPO = "/fsx/craffel/collectivedelusions/ml_research/testclaude"
results_path = os.path.join(REPO, "results", "t6_subsets.json")
tex_path = os.path.join(REPO, "paper", "paper.tex")

with open(results_path) as f:
    data = json.load(f)
agg = data["aggregate"]
scale_key = [k for k in agg if k.startswith("scale_k")][0]
sc = agg[scale_key]
ac = agg["actmat"]
gap = (sc["mean"] - ac["mean"]) * 100
mapping = {
    "T6SCAVG":  f"{sc['mean']*100:.1f}",
    "T6SCSTD":  f"{sc['std']*100:.1f}",
    "T6ACAVG":  f"{ac['mean']*100:.1f}",
    "T6ACSTD":  f"{ac['std']*100:.1f}",
    "T6GAP":    f"{gap:.1f}",
}
print(f"T=6 over {len(data['subsets'])} subsets:")
print(f"  SCALE:  {sc['mean']*100:.2f} ± {sc['std']*100:.2f}")
print(f"  ACTMat: {ac['mean']*100:.2f} ± {ac['std']*100:.2f}")
print(f"  gap:    {gap:+.2f} pp")

with open(tex_path) as f:
    tex = f.read()
for tok in sorted(mapping.keys(), key=lambda s: -len(s)):
    if tok in tex:
        tex = tex.replace(tok, mapping[tok])
        print(f"  replaced {tok} -> {mapping[tok]}")
    else:
        print(f"  WARN: {tok} not found in paper.tex")
with open(tex_path, "w") as f:
    f.write(tex)
print(f"\nPaper updated: {tex_path}")
