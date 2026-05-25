import os
import requests
import json
import re
import time

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

# Queries targeting our exact subfields
queries = [
    "model merging weight space deep learning",
    "task arithmetic neural network merging",
    "ties merging resolving interference",
    "regmean merging model weights",
    "dare model merging pruning weight",
    "git re-basin permutation symmetries",
    "zipit merging models different architectures",
    "test-time adaptation unlabeled stream",
    "tent fully test-time adaptation entropy",
    "cotta continual test-time adaptation",
    "eata efficient test-time adaptation",
    "mixture of experts routing gating network",
    "test-time model merging dynamic fusion",
    "batch normalization calibration test-time",
    "fisher information neural network merging",
    "kronecker-factored approximate curvature k-fac",
    "elastic weight consolidation continual learning",
    "parameter-efficient fine-tuning lora prefix",
    "federated learning model aggregation weights",
    "out-of-distribution generalization test-time"
]

all_bibtexs = []
seen_titles = set()
seen_keys = set()

# First read existing bibtex keys from example_paper.bib to avoid duplicating them
existing_keys = set()
if os.path.exists("example_paper.bib"):
    with open("example_paper.bib", "r") as f:
        content = f.read()
    # Find all bibtex keys
    keys = re.findall(r"@\w+{(\w+),", content)
    existing_keys.update(keys)
    print(f"Loaded {len(existing_keys)} existing keys: {existing_keys}")

for q in queries:
    print(f"Searching for: {q}...")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q.replace(' ', '+')}&fields=title,authors,year,venue,citationStyles&limit=8"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for paper in data.get("data", []):
                title = paper.get("title", "").strip().lower()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                
                citation_styles = paper.get("citationStyles", {})
                if citation_styles and "bibtex" in citation_styles:
                    bib = citation_styles["bibtex"]
                    # Extract key
                    match = re.search(r"@\w+{(\w+),", bib)
                    if match:
                        key = match.group(1)
                        if key in existing_keys or key in seen_keys:
                            continue
                        seen_keys.add(key)
                        all_bibtexs.append(bib)
        elif response.status_code == 429:
            print("Rate limited! Sleeping for 5 seconds...")
            time.sleep(5)
        else:
            print(f"Status {response.status_code} for query: {q}")
        time.sleep(1) # politely sleep
    except Exception as e:
        print(f"Error searching {q}: {e}")

print(f"Found {len(all_bibtexs)} new BibTeX entries!")

# Append to example_paper.bib
if all_bibtexs:
    with open("example_paper.bib", "a") as f:
        f.write("\n\n" + "\n\n".join(all_bibtexs))
    print("Successfully appended new entries to example_paper.bib")
else:
    print("No new entries found.")
