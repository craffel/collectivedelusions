import os
import re

bib_file = "template/example_paper.bib"
if not os.path.exists(bib_file):
    print("No bib file found")
    exit()

with open(bib_file, "r") as f:
    content = f.read()

# Find all entries and their keys, titles, authors, and years
entries = []
pattern = r"@(\w+)\s*\{\s*([^,]+),\s*author\s*=\s*\{([^}]+)\},\s*title\s*=\s*\{([^}]+)\}"
matches = re.findall(pattern, content, re.IGNORECASE)

# Let's do a simpler regex or manual parsing to be robust
current_entry = {}
entries = []

for line in content.split("\n"):
    line_stripped = line.strip()
    if line_stripped.startswith("@"):
        if current_entry and "key" in current_entry:
            entries.append(current_entry)
        current_entry = {"type": line_stripped.split("{")[0].strip("@")}
        try:
            current_entry["key"] = line_stripped.split("{")[1].split(",")[0].strip()
        except:
            current_entry = {}
    elif "author =" in line_stripped:
        try:
            current_entry["author"] = line_stripped.split("author =")[1].strip(" {},\"")
        except:
            pass
    elif "title =" in line_stripped:
        try:
            current_entry["title"] = line_stripped.split("title =")[1].strip(" {},\"")
        except:
            pass
    elif "year =" in line_stripped:
        try:
            current_entry["year"] = line_stripped.split("year =")[1].strip(" {},\"")
        except:
            pass

if current_entry and "key" in current_entry:
    entries.append(current_entry)

print(f"Parsed {len(entries)} entries manually.")

# Categorize
tta_keys = []
merging_keys = []
sharp_keys = []
sparsity_keys = []
other_keys = []

for entry in entries:
    key = entry.get("key")
    title = entry.get("title", "").lower()
    
    if "adaptation" in title or "adapt" in title or "tent" in title or "test-time" in title or "tta" in title:
        tta_keys.append(key)
    elif "merge" in title or "merg" in title or "interpolation" in title or "weight" in title or "soups" in title:
        merging_keys.append(key)
    elif "sharpness" in title or "sharp" in title or "flatness" in title or "sam" in title:
        sharp_keys.append(key)
    elif "sparse" in title or "sparsity" in title or "hoyer" in title or "gating" in title:
        sparsity_keys.append(key)
    else:
        other_keys.append(key)

print(f"Categorized keys:")
print(f"  Test-Time Adaptation: {len(tta_keys)}")
print(f"  Model Merging: {len(merging_keys)}")
print(f"  Sharpness-Aware: {len(sharp_keys)}")
print(f"  Sparsity/Gating: {len(sparsity_keys)}")
print(f"  Others: {len(other_keys)}")

# Let's save these categorization lists to json
import json
with open("classified_keys.json", "w") as f:
    json.dump({
        "tta": tta_keys,
        "merging": merging_keys,
        "sharpness": sharp_keys,
        "sparsity": sparsity_keys,
        "other": other_keys
    }, f, indent=2)
