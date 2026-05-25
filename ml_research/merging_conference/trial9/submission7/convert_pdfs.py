import os
from pypdf import PdfReader

def pdf_to_txt(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for idx, page in enumerate(reader.pages):
        text += f"--- PAGE {idx+1} ---\n"
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

papers_dir = "papers"
for file in os.listdir(papers_dir):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(papers_dir, file)
        txt_path = os.path.join(papers_dir, file.replace(".pdf", ".txt"))
        pdf_to_txt(pdf_path, txt_path)
print("Conversion complete!")
