import pypdf

def extract_pdf_text(pdf_path, txt_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- PAGE {i+1} ---\n"
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted {len(reader.pages)} pages to {txt_path}")

if __name__ == "__main__":
    extract_pdf_text("submission.pdf", "submission_text.txt")
