import pypdf
import os

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".pdf"):
        path = os.path.join(papers_dir, filename)
        print("="*40)
        print(f"File: {filename}")
        print("="*40)
        try:
            reader = pypdf.PdfReader(path)
            num_pages = len(reader.pages)
            print(f"Total pages: {num_pages}")
            # Print first 2 pages to capture Title, Abstract, and Introduction
            for i in range(min(2, num_pages)):
                print(f"--- Page {i+1} ---")
                text = reader.pages[i].extract_text()
                print(text[:3000]) # Print first 3000 chars of each page
        except Exception as e:
            print(f"Error reading {filename}: {e}")
