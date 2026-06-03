import re

print("Reading results.txt...")
try:
    with open("results.txt", "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    print("results.txt not found yet.")
    exit(1)

data = {}
for line in lines:
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 6:
        n_val = parts[0]
        method = parts[1]
        mnist = parts[2].replace("%", "")
        fmnist = parts[3].replace("%", "")
        cifar = parts[4].replace("%", "")
        avg = parts[5].replace("%", "")
        
        if n_val in ["4", "8", "16", "32", "64", "128", "256"]:
            if method == "MOMO-Merge (Shrink-Shift)":
                key = f"{n_val}-shift"
                data[key] = (mnist, fmnist, cifar, avg)
            elif method == "MOMO-Merge (Shrink-NoShift)":
                key = f"{n_val}-noshift"
                data[key] = (mnist, fmnist, cifar, avg)

print("Parsed data keys:", list(data.keys()))

print("Reading submission.tex...")
with open("submission.tex", "r") as f:
    tex = f.read()

# Replace placeholders
replacements_count = 0
for key, values in data.items():
    # Replace in table
    p1 = f"[{key}-mnist]"
    p2 = f"[{key}-fmnist]"
    p3 = f"[{key}-cifar]"
    p4 = f"[{key}-avg]"
    
    if p1 in tex:
        tex = tex.replace(p1, values[0])
        replacements_count += 1
    if p2 in tex:
        tex = tex.replace(p2, values[1])
        replacements_count += 1
    if p3 in tex:
        tex = tex.replace(p3, values[2])
        replacements_count += 1
    if p4 in tex:
        tex = tex.replace(p4, values[3])
        replacements_count += 1

print(f"Made {replacements_count} replacements in submission.tex.")

# We also replace specific placeholders in the body text:
# For example, [4-noshift-avg] can be replaced with the actual average for MOMO-Merge (Shrink-NoShift) at N=4
if "4-noshift" in data:
    tex = tex.replace("[4-noshift-avg]", data["4-noshift"][3])
if "32-shift" in data:
    tex = tex.replace("[32-shift-avg]", data["32-shift"][3])
if "256-shift" in data:
    tex = tex.replace("[256-shift-avg]", data["256-shift"][3])

with open("submission.tex", "w") as f:
    f.write(tex)

print("Done!")
