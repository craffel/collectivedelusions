import pypdf

reader = pypdf.PdfReader("submission.pdf")
print(f"Number of pages: {len(reader.pages)}")

for i, page in enumerate(reader.pages):
    print(f"--- Page {i+1} ---")
    print(page.extract_text())
