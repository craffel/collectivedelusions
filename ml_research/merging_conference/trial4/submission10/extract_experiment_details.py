import os
from pypdf import PdfReader

papers_dir = "papers"
papers = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])

for paper in papers:
    path = os.path.join(papers_dir, paper)
    print(f"\n==================================================")
    print(f"Paper: {paper}")
    print(f"==================================================")
    try:
        reader = PdfReader(path)
        # Extract pages 4, 5, 6 (0-indexed: 3, 4, 5)
        pages_to_extract = [3, 4, 5]
        for p_idx in pages_to_extract:
            if p_idx < len(reader.pages):
                print(f"--- Page {p_idx+1} ---")
                text = reader.pages[p_idx].extract_text()
                print(text[:3000]) # Print first 3000 chars of page
                print("...\n" if len(text) > 3000 else "\n")
    except Exception as e:
        print(f"Error: {e}")
