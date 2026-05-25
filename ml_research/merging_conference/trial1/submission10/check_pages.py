import pypdf

reader = pypdf.PdfReader("submission.pdf")
print("Total Pages:", len(reader.pages))
for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    if "References" in text:
        print(f"'References' found on page {idx + 1}")
