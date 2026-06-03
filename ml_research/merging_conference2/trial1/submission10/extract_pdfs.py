import pypdf
import os

def extract_pdf_info(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    
    # Extract title and abstract (usually from first few pages)
    first_page_text = reader.pages[0].extract_text()
    print(f"--- First page preview of {os.path.basename(pdf_path)} ---")
    print("\n".join(first_page_text.split("\n")[:30]))
    print("------------------------------------------------\n")
    
    full_text = []
    for i, page in enumerate(reader.pages):
        full_text.append(f"--- PAGE {i+1} ---")
        full_text.append(page.extract_text() or "")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))

os.makedirs("extracted_papers", exist_ok=True)
extract_pdf_info("papers/0.pdf", "extracted_papers/0.txt")
extract_pdf_info("papers/1.pdf", "extracted_papers/1.txt")
extract_pdf_info("papers/2.pdf", "extracted_papers/2.txt")
