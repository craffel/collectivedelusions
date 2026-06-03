import pypdf

reader = pypdf.PdfReader("submission.pdf")
print(f"Total pages: {len(reader.pages)}")

with open("submission_text.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages):
        f.write(f"--- PAGE {i+1} ---\n")
        text = page.extract_text()
        if text:
            f.write(text)
            f.write("\n")
        else:
            f.write("[No text extracted from this page]\n")
print("Extraction complete: submission_text.txt")
