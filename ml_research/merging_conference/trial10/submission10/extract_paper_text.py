import os
from pypdf import PdfReader

def extract_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

os.makedirs("papers_txt", exist_ok=True)
for filename in os.listdir("papers"):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join("papers", filename)
        txt_path = os.path.join("papers_txt", filename.replace(".pdf", ".txt"))
        extract_text(pdf_path, txt_path)
print("Done extracting all papers!")
