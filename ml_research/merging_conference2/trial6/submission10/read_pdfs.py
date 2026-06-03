import pypdf
import os

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".pdf"):
        path = os.path.join(papers_dir, filename)
        print("="*40)
        print(f"Paper: {filename}")
        print("="*40)
        reader = pypdf.PdfReader(path)
        print(f"Total pages: {len(reader.pages)}")
        # Extract first 2 pages
        for i in range(min(2, len(reader.pages))):
            print(f"--- Page {i+1} ---")
            print(reader.pages[i].extract_text()[:4000]) # Limit characters per page
            print("\n")
