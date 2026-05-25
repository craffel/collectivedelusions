import pypdf
import os

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"--- Page {i+1} ---\n"
        text += page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(reader.pages)} pages.")

if __name__ == "__main__":
    os.makedirs("papers_txt", exist_ok=True)
    extract_pdf_text("papers/submission4.pdf", "papers_txt/submission4.txt")
    extract_pdf_text("papers/submission5.pdf", "papers_txt/submission5.txt")
    extract_pdf_text("papers/submission8.pdf", "papers_txt/submission8.txt")
