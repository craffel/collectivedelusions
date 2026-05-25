import os
from pypdf import PdfReader

def inspect_paper(pdf_path):
    print("=" * 80)
    print(f"Paper: {pdf_path}")
    print("=" * 80)
    try:
        reader = PdfReader(pdf_path)
        print(f"Number of pages: {len(reader.pages)}")
        
        # Print metadata
        meta = reader.metadata
        if meta:
            print("Metadata:")
            for key, val in meta.items():
                print(f"  {key}: {val}")
        
        # Print first two pages
        print("\n--- FIRST PAGE ---")
        print(reader.pages[0].extract_text()[:3000])
        
        if len(reader.pages) > 1:
            print("\n--- SECOND PAGE ---")
            print(reader.pages[1].extract_text()[:3000])
            
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    print("\n" * 2)

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".pdf"):
        inspect_paper(os.path.join(papers_dir, filename))
