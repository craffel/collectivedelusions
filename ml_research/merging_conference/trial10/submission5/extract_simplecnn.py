import os
import re
from pypdf import PdfReader

papers_dir = "papers"
pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(papers_dir, pdf_file)
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        print(f"\n==================== {pdf_file} ====================")
        matches = [m.start() for m in re.finditer(re.compile("SimpleCNN", re.IGNORECASE), full_text)]
        for m in matches:
            start = max(0, m - 500)
            end = min(len(full_text), m + 1500)
            print("--- MATCH ---")
            print(full_text[start:end])
            print("-"*20)
    except Exception as e:
        print(f"Error: {e}")
