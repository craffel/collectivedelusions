import pypdf
import os

def extract_pages(pdf_path, start_page=1, end_page=4):
    print(f"=== {os.path.basename(pdf_path)} (Pages {start_page+1}-{end_page+1}) ===")
    try:
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        for p in range(start_page, min(end_page + 1, num_pages)):
            print(f"--- Page {p+1} ---")
            text = reader.pages[p].extract_text()
            print(text[:1500]) # Print first 1500 characters of each page
    except Exception as e:
        print(f"Error reading PDF: {e}")
    print("\n" + "="*40 + "\n")

for pdf in sorted(os.listdir("papers")):
    if pdf.endswith(".pdf"):
        extract_pages(os.path.join("papers", pdf), start_page=1, end_page=4)
