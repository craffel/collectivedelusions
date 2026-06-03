import pypdf

def extract_text_from_pdf(pdf_path, txt_path):
    print(f"Reading {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"Total pages: {num_pages}")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            f.write(f"--- PAGE {i+1} ---\n")
            text = page.extract_text()
            f.write(text)
            f.write("\n\n")
    print(f"Extracted text saved to {txt_path}")

if __name__ == "__main__":
    extract_text_from_pdf("submission.pdf", "submission_text.txt")
