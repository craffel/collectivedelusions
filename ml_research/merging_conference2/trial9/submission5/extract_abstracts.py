import pypdf
import os

def extract_first_page(pdf_path):
    print(f"=== {os.path.basename(pdf_path)} ===")
    try:
        reader = pypdf.PdfReader(pdf_path)
        if len(reader.pages) > 0:
            first_page = reader.pages[0].extract_text()
            print(first_page[:2000]) # Print first 2000 characters
        else:
            print("Empty PDF")
    except Exception as e:
        print(f"Error reading PDF: {e}")
    print("\n" + "="*40 + "\n")

for pdf in sorted(os.listdir("papers")):
    if pdf.endswith(".pdf"):
        extract_first_page(os.path.join("papers", pdf))
