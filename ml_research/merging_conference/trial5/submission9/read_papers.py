import pypdf
import os

pdf_dir = "papers"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    print("="*60)
    print(f"File: {pdf_file}")
    print("="*60)
    try:
        reader = pypdf.PdfReader(pdf_path)
        print(f"Number of pages: {len(reader.pages)}")
        # Extract first page text
        first_page = reader.pages[0]
        text = first_page.extract_text()
        print("--- Page 1 Text ---")
        print(text[:3000])  # limit to first 3000 chars of page 1
        
        # Also let's see if there is more text we want, e.g., page 2 if page 1 is short
        if len(text) < 500 and len(reader.pages) > 1:
            second_page = reader.pages[1]
            text2 = second_page.extract_text()
            print("--- Page 2 Text ---")
            print(text2[:2000])
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
    print("\n" + "="*60 + "\n")
