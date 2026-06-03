import pypdf
import os

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} -> {txt_path} ...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(text)} characters.")

for name in ["submission3", "submission7", "submission9"]:
    pdf_path = f"papers/{name}.pdf"
    txt_path = f"papers/{name}.txt"
    if os.path.exists(pdf_path):
        extract_pdf_to_txt(pdf_path, txt_path)
