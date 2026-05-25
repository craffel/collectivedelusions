import pypdf
import os

def extract_abstract(pdf_path):
    print(f"=== {pdf_path} ===")
    reader = pypdf.PdfReader(pdf_path)
    # Print first page or first two pages to find the abstract and introduction
    num_pages = len(reader.pages)
    print(f"Total pages: {num_pages}")
    
    # Extract text from first 2 pages
    text = ""
    for i in range(min(3, num_pages)):
        text += f"--- Page {i+1} ---\n"
        text += reader.pages[i].extract_text()
    
    print(text[:4000]) # Print first 4000 chars

for paper in sorted(os.listdir("papers")):
    if paper.endswith(".pdf"):
        extract_abstract(os.path.join("papers", paper))
