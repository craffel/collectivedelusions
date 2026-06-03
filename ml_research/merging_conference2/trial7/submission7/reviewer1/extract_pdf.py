import pypdf

def extract_pdf_text(pdf_path, txt_path):
    reader = pypdf.PdfReader(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, page in enumerate(reader.pages):
            f.write(f"\n--- PAGE {i + 1} ---\n")
            text = page.extract_text()
            if text:
                f.write(text)
            else:
                f.write("[No text extracted from this page]\n")
    print(f"Extracted {len(reader.pages)} pages to {txt_path}")

if __name__ == "__main__":
    extract_pdf_text("submission.pdf", "submission_text.txt")
