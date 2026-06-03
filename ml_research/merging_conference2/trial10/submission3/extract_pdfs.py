import pypdf
import os

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {i+1} ---\n" + page_text
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Finished extracting {pdf_path}.")

def main():
    papers_dir = "papers"
    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, filename)
            txt_path = os.path.join(papers_dir, filename.replace(".pdf", ".txt"))
            extract_pdf_to_txt(pdf_path, txt_path)

if __name__ == "__main__":
    main()
