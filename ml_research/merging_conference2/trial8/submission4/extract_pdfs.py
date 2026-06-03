import pypdf
import os

def extract_pdf_to_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"--- Page {idx + 1} ---\n" + page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(reader.pages)} pages.")

if __name__ == "__main__":
    os.makedirs("extracted_papers", exist_ok=True)
    extract_pdf_to_text("papers/submission3.pdf", "extracted_papers/submission3.txt")
    extract_pdf_to_text("papers/submission7.pdf", "extracted_papers/submission7.txt")
    extract_pdf_to_text("papers/submission9.pdf", "extracted_papers/submission9.txt")
