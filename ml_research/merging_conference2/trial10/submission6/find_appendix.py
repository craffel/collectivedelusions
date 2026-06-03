import pypdf
reader = pypdf.PdfReader("submission.pdf")
print("Total pages:", len(reader.pages))
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if "Appendix" in text:
        print(f"Appendix found on Page {i+1}!")
    if "Sparse Holographic" in text:
        print(f"Sparse Holographic found on Page {i+1}!")
    if "Extreme Post-Training" in text:
        print(f"Extreme Post-Training found on Page {i+1}!")
