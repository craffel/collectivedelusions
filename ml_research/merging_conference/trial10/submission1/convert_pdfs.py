import pypdf
import os

def pdf_to_txt(pdf_path, txt_path):
    print(f"Converting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text.append(f"--- Page {i+1} ---\n{page_text}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text))
    print(f"Finished {pdf_path}")

def main():
    papers_dir = "papers"
    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, filename)
            txt_path = os.path.join(papers_dir, filename.replace(".pdf", ".txt"))
            pdf_to_txt(pdf_path, txt_path)

if __name__ == "__main__":
    main()
