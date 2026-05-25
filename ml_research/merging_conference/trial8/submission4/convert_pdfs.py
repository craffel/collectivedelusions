import os
from pypdf import PdfReader

def pdf_to_txt(pdf_path, txt_path):
    print(f"Converting {pdf_path} -> {txt_path}...")
    try:
        reader = PdfReader(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                f.write(f"=== PAGE {i+1} ===\n")
                f.write(text)
                f.write("\n\n")
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    pdf_to_txt("papers/submission10.pdf", "submission10.txt")
    pdf_to_txt("papers/submission8.pdf", "submission8.txt")
    pdf_to_txt("papers/submission9.pdf", "submission9.txt")
