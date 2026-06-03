import pypdf
reader = pypdf.PdfReader("submission.pdf")
print("Total pages:", len(reader.pages))
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if "References" in text:
        print(f"References found on Page {i+1}!")
        idx = text.find("References")
        print("Context around References:")
        print(text[max(0, idx-200):idx+300])
        print("="*40)
