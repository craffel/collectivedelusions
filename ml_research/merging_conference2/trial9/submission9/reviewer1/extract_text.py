import pypdf

def extract_pdf_text(pdf_path, txt_path):
    reader = pypdf.PdfReader(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, page in enumerate(reader.pages):
            f.write(f"\n--- Page {i+1} ---\n")
            text = page.extract_text()
            if text:
                f.write(text)
                f.write("\n")
            else:
                f.write("[Empty or non-extractable page]\n")

if __name__ == "__main__":
    extract_pdf_text("submission.pdf", "submission.txt")
    print("PDF text extraction completed successfully.")
