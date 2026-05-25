import pypdf
import os

def extract_pdf_to_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- Page {i+1} ---\n"
        text += page.extract_text() + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(reader.pages)} pages.")

if __name__ == "__main__":
    for paper in ["submission8", "submission9", "submission10"]:
        pdf_path = f"papers/{paper}.pdf"
        txt_path = f"papers/{paper}.txt"
        if os.path.exists(pdf_path):
            extract_pdf_to_text(pdf_path, txt_path)
        else:
            print(f"File not found: {pdf_path}")
