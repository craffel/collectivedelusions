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
        # Search for architectural keywords
        lines = full_text.split("\n")
        for i, line in enumerate(lines):
            if any(k in line.lower() for k in ["conv", "batchnorm", "maxpool", "dropout", "kernel", "stride", "channel"]):
                print(f"Line {i}: {line.strip()}")
    except Exception as e:
        print(f"Error: {e}")
