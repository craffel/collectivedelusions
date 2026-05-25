import pypdf
import re

reader = pypdf.PdfReader("papers/0.pdf")
all_text = ""
for idx, page in enumerate(reader.pages):
    all_text += f"--- Page {idx+1} ---\n" + page.extract_text() + "\n"

# Search for matches containing "visual.proj" or "proj" or "projection layer"
matches = re.finditer(r"(visual\.proj|proj|projection layer|which layer|adapted layer|specific layer|last layer|final layer)", all_text, re.IGNORECASE)
found = []
for m in matches:
    start = max(0, m.start() - 300)
    end = min(len(all_text), m.end() + 300)
    found.append(all_text[start:end].replace("\n", " "))

print(f"Found {len(found)} references. Printing first 10:")
for f in found[:10]:
    print("-", f)
