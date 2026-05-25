import pypdf
import re

reader = pypdf.PdfReader("papers/1.pdf")
all_text = ""
for page in reader.pages:
    all_text += page.extract_text() + "\n"

# Search for Procrustes and print surrounding text
pattern = re.compile(r"Procrustes", re.IGNORECASE)
for m in pattern.finditer(all_text):
    start = max(0, m.start() - 500)
    end = min(len(all_text), m.end() + 1500)
    print(f"--- MATCH AT CHAR {m.start()} ---")
    print(all_text[start:end])
    print("\n" + "="*50 + "\n")
