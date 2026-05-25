import re

def parse_bib_keys(filepath):
    keys = set()
    try:
        with open(filepath, "r") as f:
            content = f.read()
        # Find entries like @article{key, or @inproceedings{key,
        matches = re.findall(r'@[a-zA-Z]+\{([^,\s]+),', content)
        for m in matches:
            keys.add(m.strip())
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    return keys

existing_keys = parse_bib_keys("example_paper.bib")
print(f"Existing keys: {len(existing_keys)}: {existing_keys}")

new_entries = []
seen_new_keys = set()

with open("fetched_references.bib", "r") as f:
    content = f.read()
    # Let's split by @ to isolate entries
    entries = content.split("@")
    for entry in entries:
        if not entry.strip():
            continue
        # reconstruct entry
        full_entry = "@" + entry
        # parse key
        match = re.search(r'^[a-zA-Z]+\{([^,\s]+),', entry)
        if match:
            key = match.group(1).strip()
            if key not in existing_keys and key not in seen_new_keys:
                seen_new_keys.add(key)
                new_entries.append((key, full_entry.strip()))

print(f"Parsed {len(new_entries)} unique new entries.")

# Let's take about 60 of the best new entries (which would bring our total to ~70 references)
selected_entries = new_entries[:65]

# Append them to example_paper.bib
with open("example_paper.bib", "a") as f:
    f.write("\n\n% --- ADDITIONAL RELATED WORK REFERENCES (FETCHED VIA SEMANTIC SCHOLAR) ---\n\n")
    for key, entry in selected_entries:
        f.write(entry + "\n\n")

print(f"Appended {len(selected_entries)} new references to example_paper.bib")
print("Total references in example_paper.bib should now be around", len(existing_keys) + len(selected_entries))
