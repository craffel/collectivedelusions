import pypdf
import os

def extract_info(pdf_path):
    print(f"=== {os.path.basename(pdf_path)} ===")
    try:
        reader = pypdf.PdfReader(pdf_path)
        # Extract text from the first two pages
        text = ""
        for i in range(min(2, len(reader.pages))):
            text += reader.pages[i].extract_text() + "\n"
        
        # Print first 2000 characters
        print(text[:2500])
        print("\n" + "="*40 + "\n")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

papers_dir = "papers"
for f in sorted(os.listdir(papers_dir)):
    if f.endswith(".pdf"):
        extract_info(os.path.join(papers_dir, f))
