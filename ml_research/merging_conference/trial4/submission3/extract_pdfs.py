import os
from pypdf import PdfReader

def extract_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- Page {i+1} ---\n"
        text += page.extract_text() or ""
        text += "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

papers_dir = "papers"
for filename in os.listdir(papers_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(papers_dir, filename)
        txt_path = os.path.join(papers_dir, filename[:-4] + ".txt")
        extract_text(pdf_path, txt_path)
