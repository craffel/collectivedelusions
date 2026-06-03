import os
from pypdf import PdfReader

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- PAGE {i+1} ---\n"
        text += page.extract_text() or ""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done.")

if __name__ == "__main__":
    pdf_dir = "papers"
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            txt_path = os.path.join(pdf_dir, filename.replace(".pdf", ".txt"))
            extract_pdf_text(pdf_path, txt_path)
