import pypdf
import os

def extract_full(pdf_path, txt_path):
    print(f"Extracting {pdf_path} -> {txt_path}")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open(txt_path, "w") as f:
        f.write(text)

for f in sorted(os.listdir("papers")):
    if f.endswith(".pdf"):
        extract_full(os.path.join("papers", f), f.replace(".pdf", ".txt"))
