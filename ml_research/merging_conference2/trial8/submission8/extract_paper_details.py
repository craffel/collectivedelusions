import pypdf
import os

def extract_metadata(pdf_path):
    print(f"=== {os.path.basename(pdf_path)} ===")
    try:
        reader = pypdf.PdfReader(pdf_path)
        # Extract first page text to find title and abstract
        first_page_text = reader.pages[0].extract_text()
        print("First page content (truncated):")
        print(first_page_text[:2000])
        print("\n" + "="*40 + "\n")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

papers = ["papers/submission3.pdf", "papers/submission7.pdf", "papers/submission9.pdf"]
for paper in papers:
    if os.path.exists(paper):
        extract_metadata(paper)
    else:
        print(f"File {paper} not found.")
