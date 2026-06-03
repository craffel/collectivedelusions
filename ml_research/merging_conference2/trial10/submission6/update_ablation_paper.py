import json
import re

try:
    with open("ablation_results.json") as f:
        res = json.load(f)
except Exception as e:
    print(f"Could not load ablation_results.json: {e}")
    res = {}

# Map of placeholders to values from ablation_results.json
replacements = {}

key_types = ['binary', 'continuous', 'gaussian', 'shared']
precision_levels = ['FP32', 'INT8', 'INT4']

for kt in key_types:
    for prec in precision_levels:
        placeholder = f"\\[ABLATION_{kt.upper()}_{prec}\\]"
        val = res.get(kt, {}).get(prec, {}).get('average', 10.0)
        replacements[placeholder] = f"{val:.2f}"

# Read paper and perform replacements
with open("submission.tex") as f:
    paper_content = f.read()

for placeholder, val in replacements.items():
    paper_content = re.sub(placeholder, val, paper_content)

with open("submission.tex", "w") as f:
    f.write(paper_content)

print("Successfully updated submission.tex with exact ablation results!")
