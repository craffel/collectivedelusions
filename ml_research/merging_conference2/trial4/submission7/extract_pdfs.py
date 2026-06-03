import os
from pypdf import PdfReader

def extract_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"\n--- Page {i+1} ---\n"
        page_text = page.extract_text()
        if page_text:
            text += page_text
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

os.makedirs("papers_txt", exist_ok=True)
for name in ["submission3", "submission7", "submission8"]:
    pdf_path = f"papers/{name}.pdf"
    txt_path = f"papers_txt/{name}.txt"
    if os.path.exists(pdf_path):
        extract_text(pdf_path, txt_path)
    else:
        print(f"Warning: {pdf_path} not found")
