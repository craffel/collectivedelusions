import pypdf

reader = pypdf.PdfReader("submission.pdf")
print(f"Number of pages: {len(reader.pages)}")

text = ""
for i, page in enumerate(reader.pages):
    text += f"--- Page {i+1} ---\n"
    text += page.extract_text() or ""
    text += "\n"

with open("submission.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Text extraction complete!")
