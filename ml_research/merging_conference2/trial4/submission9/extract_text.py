import pypdf
import os

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = []
    for i, page in enumerate(reader.pages):
        text.append(f"--- PAGE {i+1} ---")
        text.append(page.extract_text() or "")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))

if __name__ == "__main__":
    papers_dir = "papers"
    for file in os.listdir(papers_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, file)
            txt_path = os.path.join(papers_dir, file.replace(".pdf", ".txt"))
            extract_pdf_to_txt(pdf_path, txt_path)
