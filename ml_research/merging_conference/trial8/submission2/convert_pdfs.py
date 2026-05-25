import pypdf
import os

def pdf_to_txt(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                f.write(f"--- PAGE {i+1} ---\n")
                f.write(page.extract_text() or "")
                f.write("\n")
        print("Success.")
    except Exception as e:
        print(f"Error: {e}")

papers_dir = "papers"
for f in sorted(os.listdir(papers_dir)):
    if f.endswith(".pdf"):
        pdf_path = os.path.join(papers_dir, f)
        txt_path = f.replace(".pdf", ".txt")
        pdf_to_txt(pdf_path, txt_path)
