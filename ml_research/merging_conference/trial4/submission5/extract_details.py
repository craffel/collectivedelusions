import os
from pypdf import PdfReader

for paper in ['submission1.pdf', 'submission5.pdf', 'submission7.pdf']:
    print('='*60)
    print(paper)
    print('='*60)
    try:
        reader = PdfReader('papers/' + paper)
        num_pages = len(reader.pages)
        print(f"Total pages: {num_pages}")
        
        # Extract page 0 (usually abstract & intro)
        print("--- PAGE 0 ---")
        print(reader.pages[0].extract_text()[:2000])
        
        # Extract page 1
        print("--- PAGE 1 ---")
        print(reader.pages[1].extract_text()[:2000])
        
        # Extract the last page (conclusions or references)
        print(f"--- PAGE {num_pages-1} ---")
        print(reader.pages[num_pages-1].extract_text()[-2000:])
    except Exception as e:
        print("Error:", e)
