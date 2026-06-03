import os
from pypdf import PdfReader

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".pdf"):
        path = os.path.join(papers_dir, filename)
        print("="*60)
        print(f"Paper: {filename}")
        print("="*60)
        try:
            reader = PdfReader(path)
            print(f"Number of pages: {len(reader.pages)}")
            # Extract first page text
            first_page_text = reader.pages[0].extract_text()
            print("--- PAGE 1 ---")
            print(first_page_text[:3000]) # Print first page up to 3000 chars
            print("--- END PAGE 1 ---\n")
            if len(reader.pages) > 1:
                second_page_text = reader.pages[1].extract_text()
                print("--- PAGE 2 ---")
                print(second_page_text[:1500])
                print("--- END PAGE 2 ---\n")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
