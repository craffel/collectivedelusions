import pypdf

def extract_pdf(pdf_path, txt_path):
    reader = pypdf.PdfReader(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, page in enumerate(reader.pages):
            f.write(f"--- PAGE {idx + 1} ---\n")
            text = page.extract_text()
            if text:
                f.write(text)
            f.write("\n\n")
    print(f"Extracted {len(reader.pages)} pages to {txt_path}")

if __name__ == "__main__":
    extract_pdf("submission.pdf", "submission_extracted.txt")
