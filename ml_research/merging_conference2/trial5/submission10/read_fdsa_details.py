with open("sub10_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's find section 3 or FDSA details
import re
matches = [m.start() for m in re.finditer(r'(?:3\.\s+Methodology|Proposed Method|Fourier)', text, re.IGNORECASE)]
for m in matches:
    print(f"Match at {m}:")
    print(text[m:m+1500])
    print("-"*40)
