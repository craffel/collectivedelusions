import pypdf
import os

def extract_pages(pdf_path, out_txt_path, max_pages=4):
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(out_txt_path, "w", encoding="utf-8") as f:
            for idx in range(min(max_pages, len(reader.pages))):
                f.write(f"--- Page {idx+1} ---\n")
                f.write(reader.pages[idx].extract_text() or "")
                f.write("\n\n")
        print(f"Extracted {pdf_path} to {out_txt_path}")
    except Exception as e:
        print(f"Error {pdf_path}: {e}")

os.makedirs("extracted_papers", exist_ok=True)
extract_pages("papers/0.pdf", "extracted_papers/0.txt")
extract_pages("papers/1.pdf", "extracted_papers/1.txt")
extract_pages("papers/2.pdf", "extracted_papers/2.txt")
