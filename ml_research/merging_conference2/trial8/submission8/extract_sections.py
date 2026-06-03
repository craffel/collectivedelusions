import pypdf
import os
import re

def search_pdf(pdf_path):
    print(f"\n==================== {os.path.basename(pdf_path)} ====================")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    
    # Let's extract section headers (lines that look like headers, e.g., "1. Introduction", "3. Method", etc.)
    # and any Algorithm boxes or Python code blocks.
    all_text = ""
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        all_text += f"\n--- Page {idx+1} ---\n" + text
        
    # Search for headings
    headings = re.findall(r'\n(Section \d+|[1-9]\.?\s+[A-Z][a-z]+.*?)\n', all_text)
    print("Detected Headings/Key Lines:")
    for h in headings[:15]:
        print(" -", h.strip())
        
    # Search for "Algorithm"
    print("\n--- Algorithm Mentions ---")
    lines = all_text.split('\n')
    for line in lines:
        if 'Algorithm' in line or 'PyTorch' in line or 'def ' in line:
            print(line[:120])

    # Save the full extracted text of the papers to text files so we can grep them easily if needed
    txt_path = pdf_path.replace('.pdf', '.txt')
    with open(txt_path, 'w') as f:
        f.write(all_text)
    print(f"Saved full text to {txt_path}")

papers = ["papers/submission3.pdf", "papers/submission7.pdf", "papers/submission9.pdf"]
for paper in papers:
    if os.path.exists(paper):
        search_pdf(paper)
