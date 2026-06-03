import pypdf
import os

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"--- Page {page_num + 1} ---\n"
        text += page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(reader.pages)} pages.")

if __name__ == "__main__":
    papers_dir = "papers"
    for file in os.listdir(papers_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, file)
            txt_path = os.path.join(papers_dir, file.replace(".pdf", ".txt"))
            extract_pdf_text(pdf_path, txt_path)
