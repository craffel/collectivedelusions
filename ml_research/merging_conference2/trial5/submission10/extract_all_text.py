import pypdf
import re

def extract_sections(pdf_path, output_txt_path):
    print(f"Extracting sections from {pdf_path} to {output_txt_path}")
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for idx, page in enumerate(reader.pages):
        full_text += f"\n--- Page {idx+1} ---\n" + page.extract_text()
        
    # Write full text
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    # Print some info
    print(f"Total pages: {len(reader.pages)}, Total chars: {len(full_text)}")
    # Find headings
    headings = re.findall(r'\n(?:[0-9]+\.?\s+[A-Z][A-Za-z ]+)\n', full_text)
    print("Found headings:", headings)

extract_sections("papers/submission3.pdf", "sub3_text.txt")
extract_sections("papers/submission9.pdf", "sub9_text.txt")
extract_sections("papers/submission10.pdf", "sub10_text.txt")
