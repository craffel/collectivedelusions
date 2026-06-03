import pypdf

reader = pypdf.PdfReader("submission.pdf")
print(f"Number of pages: {len(reader.pages)}")

with open("submission.txt", "w", encoding="utf-8") as f:
    for idx, page in enumerate(reader.pages):
        f.write(f"\n--- PAGE {idx + 1} ---\n")
        text = page.extract_text()
        if text:
            f.write(text)
        else:
            f.write("[No text extracted from this page]\n")

print("Extraction completed successfully.")
