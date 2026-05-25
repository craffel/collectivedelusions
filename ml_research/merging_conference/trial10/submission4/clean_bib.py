import os

bib_file = "template/example_paper.bib"
if not os.path.exists(bib_file):
    print("No bib file found")
    exit()

with open(bib_file, "r") as f:
    content = f.read()

print(f"Original content length: {len(content)}")

# Replace unicode characters or keys that might cause issues
replacements = {
    "koikeakino2025μmoe": "koikeakino2025mumoe",
    "μ-MoE": r"$\mu$-MoE",
    "μ": r"$\mu$",
    "’": "'",
    "“": "``",
    "”": "''",
    "–": "--",
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Let's fix keys in the file if they contain μ
content = content.replace("koikeakino2025\u03bcmoe", "koikeakino2025mumoe")

# Escape unescaped underscores in title or note, but NOT in keys or field names
# We can do this carefully line-by-line
new_lines = []
for line in content.split("\n"):
    if line.strip().startswith("@"):
        new_lines.append(line)
        continue
    # If the line contains an underscore and it's not escaped
    if "_" in line and r"\_" not in line:
        # Check if it's a DOI or key
        if "doi =" in line or "url =" in line or "file =" in line:
            new_lines.append(line)
        else:
            line = line.replace("_", r"\_")
            new_lines.append(line)
    else:
        new_lines.append(line)

content = "\n".join(new_lines)

# Write cleaned bib back
with open(bib_file, "w") as f:
    f.write(content)

print("Finished cleaning template/example_paper.bib")
