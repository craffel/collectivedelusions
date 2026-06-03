import pypdf

def extract_pdf_text(pdf_path, txt_path):
    print(f"Reading {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = []
    for i, page in enumerate(reader.pages):
        text.append(f"--- PAGE {i+1} ---")
        text.append(page.extract_text())
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print(f"Extracted text saved to {txt_path}")

if __name__ == "__main__":
    extract_pdf_text("submission.pdf", "submission_text.txt")
