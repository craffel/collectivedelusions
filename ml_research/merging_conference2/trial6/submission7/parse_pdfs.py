import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"--- PAGE {i+1} ---\n" + page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(reader.pages)} pages.")

if __name__ == "__main__":
    os.makedirs("papers_txt", exist_ok=True)
    extract_text_from_pdf("papers/submission1.pdf", "papers_txt/submission1.txt")
    extract_text_from_pdf("papers/submission7.pdf", "papers_txt/submission7.txt")
    extract_text_from_pdf("papers/submission9.pdf", "papers_txt/submission9.txt")
