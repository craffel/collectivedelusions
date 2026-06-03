import pypdf
import sys

def extract_pdf_pages(pdf_path, max_pages=3):
    print(f"=== Extracting {pdf_path} (up to {max_pages} pages) ===")
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    for i in range(min(max_pages, num_pages)):
        print(f"--- Page {i+1} ---")
        text = reader.pages[i].extract_text()
        print(text[:2000]) # Print first 2000 chars of page

if __name__ == '__main__':
    if len(sys.argv) > 1:
        extract_pdf_pages(sys.argv[1])
    else:
        print("Please provide a PDF path.")
