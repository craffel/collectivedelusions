import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"--- Page {i+1} ---\n" + page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted {len(reader.pages)} pages.")

for i in [1, 3, 6]:
    pdf_file = f"papers/submission{i}.pdf"
    txt_file = f"papers/submission{i}.txt"
    if os.path.exists(pdf_file):
        extract_text_from_pdf(pdf_file, txt_file)
    else:
        print(f"File not found: {pdf_file}")
