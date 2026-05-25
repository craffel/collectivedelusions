import pypdf

reader = pypdf.PdfReader('submission.pdf')
print(f"Total pages: {len(reader.pages)}")
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    first_lines = "\n".join(text.splitlines()[:3])
    print(f"--- Page {i+1} ---")
    print(first_lines)
    print("...")
