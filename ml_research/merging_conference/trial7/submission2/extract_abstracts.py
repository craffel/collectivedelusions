import pypdf
import os

pdf_dir = "papers"
pdf_files = ["submission3.pdf", "submission6.pdf", "submission10.pdf"]

for f in pdf_files:
    path = os.path.join(pdf_dir, f)
    print("="*40)
    print(f"File: {f}")
    print("="*40)
    try:
        reader = pypdf.PdfReader(path)
        print(f"Number of pages: {len(reader.pages)}")
        # Print text from first 2 pages
        for i in range(min(2, len(reader.pages))):
            print(f"--- Page {i+1} ---")
            print(reader.pages[i].extract_text()[:4000]) # Limit to 4k chars per page
    except Exception as e:
        print(f"Error reading {f}: {e}")
