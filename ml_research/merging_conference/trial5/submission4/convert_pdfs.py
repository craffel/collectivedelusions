import os
from pypdf import PdfReader

def pdf_to_text(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        text += f"--- Page {page_num + 1} ---\n"
        text += page.extract_text() + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Finished {pdf_path}")

pdf_to_text("papers/submission6.pdf", "papers/submission6.txt")
pdf_to_text("papers/submission8.pdf", "papers/submission8.txt")
pdf_to_text("papers/submission9.pdf", "papers/submission9.txt")
