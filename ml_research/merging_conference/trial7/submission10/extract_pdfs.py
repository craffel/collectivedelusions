import pypdf
import os

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"--- PAGE {i+1} ---\n" + page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(reader.pages)} pages.")

if __name__ == "__main__":
    os.makedirs("papers", exist_ok=True)
    extract_pdf_to_txt("papers/submission10.pdf", "papers/submission10.txt")
    extract_pdf_to_txt("papers/submission3.pdf", "papers/submission3.txt")
    extract_pdf_to_txt("papers/submission6.pdf", "papers/submission6.txt")
