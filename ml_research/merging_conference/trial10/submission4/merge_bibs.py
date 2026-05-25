import os

existing_file = "template/example_paper.bib"
fetched_file = "fetched_papers.bib"

existing_content = ""
if os.path.exists(existing_file):
    with open(existing_file, "r") as f:
        existing_content = f.read()

fetched_content = ""
if os.path.exists(fetched_file):
    with open(fetched_file, "r") as f:
        fetched_content = f.read()

# Parse existing keys to avoid duplicating keys
existing_keys = set()
for line in existing_content.split("\n"):
    line = line.strip()
    if line.startswith("@") and "{" in line:
        try:
            key = line.split("{")[1].split(",")[0].strip()
            existing_keys.add(key)
        except Exception:
            pass

print(f"Existing keys: {len(existing_keys)}")

# Let's read the fetched entries and filter out duplicates
fetched_entries = []
current_entry = []
current_key = None
is_inside = False

for line in fetched_content.split("\n"):
    line_stripped = line.strip()
    if line_stripped.startswith("@") and "{" in line_stripped:
        if current_entry:
            # save previous entry
            fetched_entries.append((current_key, "\n".join(current_entry)))
            current_entry = []
        try:
            current_key = line_stripped.split("{")[1].split(",")[0].strip()
        except Exception:
            current_key = None
        is_inside = True
    current_entry.append(line)

if current_entry and current_key:
    fetched_entries.append((current_key, "\n".join(current_entry)))

filtered_fetched = []
for key, entry in fetched_entries:
    if key not in existing_keys:
        filtered_fetched.append(entry)
        existing_keys.add(key)

print(f"New unique keys added: {len(filtered_fetched)}")

merged_content = existing_content + "\n\n" + "\n\n".join(filtered_fetched)

with open(existing_file, "w") as f:
    f.write(merged_content)

print(f"Successfully wrote merged bib entries to {existing_file}")
