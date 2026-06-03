import pypdf

def extract_text(pdf_path, txt_path):
    print(f"Reading {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            f.write(f"--- PAGE {i+1} ---\n")
            f.write(text)
            f.write("\n\n")
    print(f"Text successfully extracted to {txt_path}")

if __name__ == "__main__":
    extract_text("submission.pdf", "submission.txt")
