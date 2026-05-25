import pypdf

reader = pypdf.PdfReader("submission.pdf")
print(f"Number of pages: {len(reader.pages)}")

with open("submission_text.txt", "w") as f:
    for i, page in enumerate(reader.pages):
        f.write(f"\n--- PAGE {i+1} ---\n")
        f.write(page.extract_text())

print("Successfully extracted all pages to submission_text.txt")
