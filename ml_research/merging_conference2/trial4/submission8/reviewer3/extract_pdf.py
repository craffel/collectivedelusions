import pypdf

def extract_pdf_text(pdf_path, txt_path):
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"Total pages: {num_pages}")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, page in enumerate(reader.pages):
            f.write(f"\n--- Page {idx+1} ---\n")
            text = page.extract_text()
            if text:
                f.write(text)
            else:
                f.write("[No text extracted from this page]\n")

if __name__ == "__main__":
    extract_pdf_text("submission.pdf", "submission_text.txt")
