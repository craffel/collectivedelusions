import os
from pypdf import PdfReader

def pdf_to_text(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        text += f"--- Page {page_num + 1} ---\n"
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    papers_dir = "papers"
    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, filename)
            txt_path = os.path.join(papers_dir, filename.replace(".pdf", ".txt"))
            pdf_to_text(pdf_path, txt_path)

if __name__ == "__main__":
    main()
