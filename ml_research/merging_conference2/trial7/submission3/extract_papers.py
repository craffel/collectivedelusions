import pypdf
import os

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"--- PAGE {i+1} ---\n" + page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(reader.pages)} pages.")

if __name__ == "__main__":
    for paper in ["submission5", "submission6", "submission10"]:
        pdf_path = f"papers/{paper}.pdf"
        txt_path = f"papers/{paper}.txt"
        if os.path.exists(pdf_path):
            extract_pdf_to_txt(pdf_path, txt_path)
        else:
            print(f"File not found: {pdf_path}")
