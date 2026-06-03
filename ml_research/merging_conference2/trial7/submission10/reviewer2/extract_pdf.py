import pypdf

def extract_pdf_to_text(pdf_path, txt_path):
    print(f"Reading {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            f.write(f"\n--- Page {i + 1} ---\n")
            text = page.extract_text()
            if text:
                f.write(text)
            else:
                f.write("[Empty or non-extractable page]\n")
    print(f"Extraction complete! Saved to {txt_path}.")

if __name__ == "__main__":
    extract_pdf_to_text("submission.pdf", "submission.txt")
