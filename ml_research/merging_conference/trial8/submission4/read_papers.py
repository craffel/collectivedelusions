import os
from pypdf import PdfReader

def extract_first_pages(pdf_path, max_pages=3):
    print("="*60)
    print(f"Reading: {pdf_path}")
    print("="*60)
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"Total pages: {num_pages}")
        pages_to_read = min(max_pages, num_pages)
        for i in range(pages_to_read):
            text = reader.pages[i].extract_text()
            print(f"--- Page {i+1} ---")
            print(text[:3000]) # Print first 3000 chars of each page
            print("\n")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

if __name__ == "__main__":
    papers_dir = "papers"
    papers = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])
    for paper in papers:
        extract_first_pages(os.path.join(papers_dir, paper))
