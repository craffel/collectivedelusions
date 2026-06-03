import os
from pypdf import PdfReader

def extract_first_pages(pdf_path, num_pages=2):
    print("="*80)
    print(f"Extracting {pdf_path}")
    print("="*80)
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} does not exist.")
        return
    try:
        reader = PdfReader(pdf_path)
        num_pages = min(num_pages, len(reader.pages))
        for i in range(num_pages):
            print(f"--- Page {i+1} ---")
            print(reader.pages[i].extract_text()[:4000]) # Print first 4000 chars of page
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

papers = ["papers/submission1.pdf", "papers/submission2.pdf", "papers/submission9.pdf"]
for paper in papers:
    extract_first_pages(paper)
