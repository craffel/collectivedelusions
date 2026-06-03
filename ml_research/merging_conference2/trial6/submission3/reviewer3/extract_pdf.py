import pypdf

def extract_pdf_text(pdf_path, txt_path):
    reader = pypdf.PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, page in enumerate(reader.pages):
            f.write(f"--- PAGE {i+1} ---\n")
            f.write(page.extract_text() or "")
            f.write("\n")

if __name__ == "__main__":
    extract_pdf_text("submission.pdf", "submission_text.txt")
    print("Done extracting PDF text.")
