import os
from pypdf import PdfReader

papers_dir = "papers"
pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(papers_dir, pdf_file)
    print(f"=== File: {pdf_file} ===")
    try:
        reader = PdfReader(pdf_path)
        print(f"Number of pages: {len(reader.pages)}")
        
        # Extract first 2 pages (usually contains title, abstract, and introduction)
        text = ""
        for i in range(min(2, len(reader.pages))):
            text += f"--- PAGE {i+1} ---\n"
            text += reader.pages[i].extract_text() + "\n"
        
        print(text[:4000]) # Limit print size
        print("="*40 + "\n")
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}\n")
