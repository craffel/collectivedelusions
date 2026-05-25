import os
import re
from pypdf import PdfReader

papers_dir = "papers"
pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(papers_dir, pdf_file)
    print(f"\n==================== {pdf_file} ====================")
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        # Search for github links, websites, datasets
        urls = re.findall(r'https?://\S+', full_text)
        print("URLs found:")
        for url in set(urls):
            print("  ", url)
            
        print("\nDatasets or datasets-related terms found:")
        lines = full_text.split('\n')
        for i, line in enumerate(lines):
            if any(term in line.lower() for term in ["dataset", "cifar", "mnist", "imagenet", "benchmark"]):
                print(f"  Line {i}: {line.strip()}")
                
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
