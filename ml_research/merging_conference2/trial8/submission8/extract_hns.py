import re

with open("papers/submission7.txt", "r") as f:
    text = f.read()

print("=== Search for HNS formulas and algorithms in submission7 ===")
# Search for HNS equations, formulas, or "Algorithm" in submission7.txt
matches = [m.start() for m in re.finditer(r"HNS|Holographic Norm", text, re.IGNORECASE)]
for m in matches[:10]:
    print(text[max(0, m-200):min(len(text), m+1000)])
    print("-"*50)
