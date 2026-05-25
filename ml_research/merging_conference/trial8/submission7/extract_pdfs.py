import os
from pypdf import PdfReader

def pdf_to_txt(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            f.write(f"--- PAGE {i+1} ---\n")
            f.write(text)
            f.write("\n")

for name in ["submission8", "submission9", "submission10"]:
    pdf_path = f"papers/{name}.pdf"
    txt_path = f"papers/{name}.txt"
    if os.path.exists(pdf_path):
        pdf_to_txt(pdf_path, txt_path)
    else:
        print(f"Warning: {pdf_path} not found.")
