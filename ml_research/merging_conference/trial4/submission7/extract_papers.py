import sys
from pypdf import PdfReader

def extract_pdf_preview(pdf_path, txt_path):
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"Total pages: {num_pages}")
    
    # Extract all text to txt_path
    full_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        full_text.append(f"--- PAGE {i+1} ---")
        full_text.append(text)
        
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))
    print(f"Saved full text to {txt_path}")

    # Print first page preview
    print(f"--- First 1000 chars of {pdf_path} ---")
    first_page_text = reader.pages[0].extract_text()
    print(first_page_text[:1000])
    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    extract_pdf_preview("papers/submission1.pdf", "papers/submission1.txt")
    extract_pdf_preview("papers/submission5.pdf", "papers/submission5.txt")
    extract_pdf_preview("papers/submission7.pdf", "papers/submission7.txt")
