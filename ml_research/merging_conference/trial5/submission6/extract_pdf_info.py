import pypdf
import os

pdf_dir = "papers"
for pdf_file in sorted(os.listdir(pdf_dir)):
    if pdf_file.endswith(".pdf"):
        path = os.path.join(pdf_dir, pdf_file)
        print("="*60)
        print(f"File: {pdf_file}")
        print("="*60)
        try:
            reader = pypdf.PdfReader(path)
            num_pages = len(reader.pages)
            print(f"Number of pages: {num_pages}")
            
            # Extract first 2 pages for Title and Abstract
            first_pages_text = ""
            for i in range(min(2, num_pages)):
                first_pages_text += reader.pages[i].extract_text() + "\n"
            
            # Just print the first 2500 characters
            print(first_pages_text[:3000])
            print("\n" + "-"*40 + "\n")
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
