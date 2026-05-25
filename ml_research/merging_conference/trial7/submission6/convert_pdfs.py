import os
from pypdf import PdfReader

def pdf_to_text(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

os.makedirs("papers_txt", exist_ok=True)
pdf_to_text("papers/submission3.pdf", "papers_txt/submission3.txt")
pdf_to_text("papers/submission6.pdf", "papers_txt/submission6.txt")
pdf_to_text("papers/submission10.pdf", "papers_txt/submission10.txt")
print("All papers converted successfully.")
