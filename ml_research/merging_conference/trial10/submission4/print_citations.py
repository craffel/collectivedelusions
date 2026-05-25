import os
import json

bib_file = "template/example_paper.bib"
with open(bib_file, "r") as f:
    content = f.read()

# Let's extract all keys and their titles using a simple state machine
entries = []
current_entry = {}

for line in content.split("\n"):
    line_stripped = line.strip()
    if line_stripped.startswith("@"):
        if current_entry and "key" in current_entry:
            entries.append(current_entry)
        current_entry = {}
        try:
            current_entry["key"] = line_stripped.split("{")[1].split(",")[0].strip()
        except:
            pass
    elif "title =" in line_stripped:
        try:
            current_entry["title"] = line_stripped.split("title =")[1].strip(" {},\"")
        except:
            pass
    elif "author =" in line_stripped:
        try:
            current_entry["author"] = line_stripped.split("author =")[1].strip(" {},\"")
        except:
            pass

if current_entry and "key" in current_entry:
    entries.append(current_entry)

print(f"Total entries parsed: {len(entries)}")

# Classify and filter
categories = {
    "tta": {
        "keywords": ["adapt", "tent", "tta", "test-time", "streaming", "domain adaptation", "unsupervised", "shifter", "entropy"],
        "keys": []
    },
    "merging": {
        "keywords": ["merg", "soup", "interpolat", "arithmetic", "ties", "fuse", "fusion", "blend"],
        "keys": []
    },
    "sharpness": {
        "keywords": ["sharp", "sam", "flat", "minima", "regulariz", "geometry"],
        "keys": []
    },
    "sparsity": {
        "keywords": ["spars", "hoyer", "gate", "gating", "threshold", "zero", "activation", "compression", "mask", "mixture of experts", "moe"],
        "keys": []
    }
}

for entry in entries:
    key = entry.get("key")
    if not key:
        continue
    title = entry.get("title", "").lower()
    
    # check each category
    for cat_name, cat in categories.items():
        if any(kw in title or kw in key.lower() for kw in cat["keywords"]):
            cat["keys"].append(key)

for cat_name, cat in categories.items():
    unique_keys = sorted(list(set(cat["keys"])))
    print(f"\nCategory: {cat_name.upper()} ({len(unique_keys)} keys found)")
    # Print in groups of 12 keys for easy pasting
    for i in range(0, len(unique_keys), 12):
        chunk = unique_keys[i:i+12]
        print(f"  \\cite{{{', '.join(chunk)}}}")
