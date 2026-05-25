import pypdf
import glob
import os

for path in sorted(glob.glob("papers/*.pdf")):
    print("="*40)
    print(f"File: {path}")
    print("="*40)
    try:
        reader = pypdf.PdfReader(path)
        print(f"Number of pages: {len(reader.pages)}")
        # Extract first page text to get title/abstract
        first_page = reader.pages[0].extract_text()
        print("--- FIRST PAGE ---")
        print(first_page[:2000]) # Print first 2000 chars of first page
        if len(reader.pages) > 1:
            print("--- SECOND PAGE (START) ---")
            print(reader.pages[1].extract_text()[:1000])
    except Exception as e:
        print(f"Error reading {path}: {e}")
