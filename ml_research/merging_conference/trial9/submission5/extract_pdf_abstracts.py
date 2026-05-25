import pypdf
import os

def extract_first_pages(pdf_path, num_pages=2):
    print("="*80)
    print(f"File: {pdf_path}")
    print("="*80)
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i in range(min(num_pages, len(reader.pages))):
            print(f"--- Page {i+1} ---")
            print(reader.pages[i].extract_text()[:3000]) # Print first 3000 chars per page
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    print("\n" + "="*80 + "\n")

for name in sorted(os.listdir("papers")):
    if name.endswith(".pdf"):
        extract_first_pages(os.path.join("papers", name))
