import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, txt_path):
    print(f"Extracting {pdf_path} -> {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"--- Page {page_num + 1} ---\n" + page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted {len(reader.pages)} pages, {len(text)} characters.")
    
    # Print the first 1000 characters of the text
    print(f"--- First 1000 characters of {pdf_path} ---")
    print(text[:1000])
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    papers_dir = "papers"
    for filename in sorted(os.listdir(papers_dir)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, filename)
            txt_path = filename.replace(".pdf", ".txt")
            extract_text_from_pdf(pdf_path, txt_path)
