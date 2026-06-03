import os
from pypdf import PdfReader

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".pdf"):
        path = os.path.join(papers_dir, filename)
        print(f"=== {filename} ===")
        try:
            reader = PdfReader(path)
            print(f"Number of pages: {len(reader.pages)}")
            # Print first 2 pages
            for i in range(min(2, len(reader.pages))):
                print(f"--- Page {i+1} ---")
                print(reader.pages[i].extract_text()[:4000]) # up to 4000 chars per page
        except Exception as e:
            print(f"Error reading {filename}: {e}")
        print("\n" + "="*40 + "\n")
