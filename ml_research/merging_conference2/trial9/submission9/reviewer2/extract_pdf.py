import pypdf

reader = pypdf.PdfReader("submission.pdf")
print(f"Number of pages: {len(reader.pages)}")

text = []
for i, page in enumerate(reader.pages):
    page_text = page.extract_text()
    text.append(f"--- PAGE {i+1} ---")
    text.append(page_text)

with open("submission.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(text))

print("Extracted submission.pdf to submission.txt")
