import pypdf

def extract_text(pdf_path, txt_path):
    reader = pypdf.PdfReader(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, page in enumerate(reader.pages):
            f.write(f"--- PAGE {i+1} ---\n")
            text = page.extract_text()
            f.write(text)
            f.write("\n\n")

if __name__ == "__main__":
    extract_text("submission.pdf", "submission.txt")
    print("PDF extraction completed successfully.")
