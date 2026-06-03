import os
from pypdf import PdfReader

def extract_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = PdfReader(pdf_path)
    text_content = []
    for i, page in enumerate(reader.pages):
        text_content.append(f"--- Page {i+1} ---")
        text_content.append(page.extract_text() or "")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_content))

if __name__ == "__main__":
    papers_dir = "papers"
    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, filename)
            txt_path = os.path.join(papers_dir, filename.replace(".pdf", ".txt"))
            extract_text(pdf_path, txt_path)
    print("Done!")
