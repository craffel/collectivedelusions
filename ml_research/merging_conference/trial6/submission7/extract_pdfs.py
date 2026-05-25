import pypdf
import os

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- Page {i+1} ---\n"
        text += page.extract_text() or ""
        text += "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted {len(reader.pages)} pages.")

os.makedirs("papers", exist_ok=True)
extract_pdf_to_txt("papers/submission2.pdf", "papers/submission2.txt")
extract_pdf_to_txt("papers/submission7.pdf", "papers/submission7.txt")
extract_pdf_to_txt("papers/submission8.pdf", "papers/submission8.txt")
