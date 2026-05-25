import pypdf
import os

def extract_full_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} -> {txt_path}")
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                f.write(f"\n--- Page {i+1} ---\n")
                f.write(page.extract_text() or "")
    except Exception as e:
        print(f"Error: {e}")

for name in sorted(os.listdir("papers")):
    if name.endswith(".pdf"):
        extract_full_text(os.path.join("papers", name), os.path.join("papers", name.replace(".pdf", ".txt")))
