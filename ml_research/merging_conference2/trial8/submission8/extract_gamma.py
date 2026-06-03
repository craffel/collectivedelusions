import re

with open("papers/submission7.txt", "r") as f:
    text = f.read()

print("=== Search for gamma or channel-wise scaling formulation ===")
matches = [m.start() for m in re.finditer(r"channel-wise|gamma|scale|Formula|Equation|Algorithm", text, re.IGNORECASE)]
seen = set()
for m in matches:
    # Get paragraph around match
    start = max(0, m-100)
    end = min(len(text), m+300)
    snippet = text[start:end]
    if snippet[:100] not in seen:
        seen.add(snippet[:100])
        print(snippet)
        print("-"*50)
