import os
import pypdf

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- Page {i+1} ---\n"
        text += page.extract_text() or ""
        text += "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

os.makedirs("extracted_papers", exist_ok=True)
extract_pdf_text("papers/submission3.pdf", "extracted_papers/submission3.txt")
extract_pdf_text("papers/submission7.pdf", "extracted_papers/submission7.txt")
extract_pdf_text("papers/submission9.pdf", "extracted_papers/submission9.txt")
print("Done!")
