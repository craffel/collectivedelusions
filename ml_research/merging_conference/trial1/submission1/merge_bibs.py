import os

def parse_bib_keys(file_path):
    keys = set()
    if not os.path.exists(file_path):
        return keys
    with open(file_path, "r") as f:
        for line in f:
            if line.strip().startswith("@"):
                # extract key
                try:
                    parts = line.split("{", 1)
                    if len(parts) > 1:
                        key = parts[1].split(",", 1)[0].strip()
                        keys.add(key)
                except Exception:
                    pass
    return keys

orig_keys = parse_bib_keys("submission.bib")
print(f"Original keys ({len(orig_keys)}):", orig_keys)

# Read original file content
with open("submission.bib", "r") as f:
    orig_content = f.read()

# Let's read fetched_papers.bib and extract entries that don't duplicate keys
fetched_entries = []
current_entry = []
inside_entry = False
current_key = None

with open("fetched_papers.bib", "r") as f:
    for line in f:
        stripped = line.strip()
        if stripped.startswith("@"):
            inside_entry = True
            current_entry = [line]
            try:
                parts = line.split("{", 1)
                if len(parts) > 1:
                    current_key = parts[1].split(",", 1)[0].strip()
            except Exception:
                current_key = None
        elif inside_entry:
            current_entry.append(line)
            if stripped == "}":
                inside_entry = False
                if current_key and current_key not in orig_keys:
                    fetched_entries.append((current_key, "".join(current_entry)))
                current_key = None

print(f"Parsed {len(fetched_entries)} fetched non-duplicate entries.")

# Select the top ones to reach at least 55 total references (original 11 + 44 new)
needed = 55 - len(orig_keys)
selected = fetched_entries[:max(needed, 45)]

print(f"Selected {len(selected)} new entries.")

# Append to submission.bib
with open("submission.bib", "a") as f:
    f.write("\n\n% --- AUTOMATICALLY ADDED RELATED WORK BIBLIOGRAPHY ---\n\n")
    for key, entry in selected:
        f.write(entry + "\n\n")

print("Updated submission.bib successfully!")
