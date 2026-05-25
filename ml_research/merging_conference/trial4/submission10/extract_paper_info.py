import os
from pypdf import PdfReader

papers_dir = "papers"
papers = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]

print(f"Found papers: {papers}\n")

for paper in sorted(papers):
    path = os.path.join(papers_dir, paper)
    print(f"=== Reading {paper} ===")
    try:
        reader = PdfReader(path)
        num_pages = len(reader.pages)
        print(f"Number of pages: {num_pages}")
        
        # Extract first 2 pages to get Title, Abstract, Introduction
        text = ""
        for i in range(min(2, num_pages)):
            text += f"--- Page {i+1} ---\n"
            text += reader.pages[i].extract_text()
            text += "\n"
        
        # Print a snippet of the beginning
        print(text[:3000])
        print("...\n" if len(text) > 3000 else "\n")
    except Exception as e:
        print(f"Error reading {paper}: {e}\n")
