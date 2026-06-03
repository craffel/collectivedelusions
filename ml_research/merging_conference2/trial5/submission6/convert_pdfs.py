import pypdf
import os
import sys

def convert_pdf_to_txt(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"\n--- PAGE {i+1} ---\n"
        text += page.extract_text()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    papers_dir = "papers"
    for file in os.listdir(papers_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, file)
            txt_path = os.path.join(papers_dir, file.replace(".pdf", ".txt"))
            try:
                convert_pdf_to_txt(pdf_path, txt_path)
            except Exception as e:
                print(f"Error converting {pdf_path}: {e}")

if __name__ == "__main__":
    main()
