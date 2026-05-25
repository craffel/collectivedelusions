import pypdf
import os

def pdf_to_txt(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for idx, page in enumerate(reader.pages):
        text += f"\n\n--- Page {idx+1} ---\n\n"
        text += page.extract_text()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

os.makedirs("papers_txt", exist_ok=True)
for paper in sorted(os.listdir("papers")):
    if paper.endswith(".pdf"):
        pdf_to_txt(os.path.join("papers", paper), os.path.join("papers_txt", paper.replace(".pdf", ".txt")))
