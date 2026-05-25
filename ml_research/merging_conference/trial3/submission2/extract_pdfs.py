import pypdf
import os

def pdf_to_text(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"--- Page {i+1} ---\n" + (page_text if page_text else "") + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(text)} characters.")

papers_dir = "papers"
for filename in os.listdir(papers_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(papers_dir, filename)
        txt_path = os.path.join(papers_dir, filename[:-4] + ".txt")
        pdf_to_text(pdf_path, txt_path)
