import pypdf
import os

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} -> {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- Page {i+1} ---\n"
        text += page.extract_text() or ""
        text += "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

os.makedirs("papers", exist_ok=True)
extract_pdf_text("papers/submission1.pdf", "papers/submission1.txt")
extract_pdf_text("papers/submission5.pdf", "papers/submission5.txt")
extract_pdf_text("papers/submission7.pdf", "papers/submission7.txt")
print("Done!")
