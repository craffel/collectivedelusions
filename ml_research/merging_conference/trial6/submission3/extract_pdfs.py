import pypdf
import os

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} -> {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"--- PAGE {i+1} ---\n" + page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted {len(reader.pages)} pages.")

for i in [2, 7, 8]:
    pdf_path = f"papers/submission{i}.pdf"
    txt_path = f"papers/submission{i}.txt"
    if os.path.exists(pdf_path):
        extract_pdf_to_txt(pdf_path, txt_path)
    else:
        print(f"PDF {pdf_path} not found.")
