from pypdf import PdfReader

reader = PdfReader("papers/submission4.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

lines = full_text.split("\n")
for idx in range(min(710, len(lines)), min(770, len(lines))):
    print(f"{idx}: {lines[idx]}")
