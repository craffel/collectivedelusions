with open("sata_details.txt") as f:
    text = f.read()

import re

# Search for equations or terms
keywords = ["rgp", "sbf", "equation", "formula", "convex", "fisher"]
for kw in keywords:
    print(f"\n=== OCCURRENCES OF '{kw.upper()}' ===")
    matches = [m.start() for m in re.finditer(kw, text, re.IGNORECASE)]
    print(f"Found {len(matches)} matches.")
    for m in matches[:5]: # print first 5 matches with context
        start = max(0, m - 200)
        end = min(len(text), m + 400)
        print(f"--- MATCH AT {m} ---")
        print(text[start:end])
        print("*" * 40)
