import pypdf

def extract_text(pdf_path, txt_path):
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, page in enumerate(reader.pages):
            f.write(f"--- PAGE {idx+1} ---\n")
            text = page.extract_text()
            if text:
                f.write(text)
                f.write("\n")
            else:
                print(f"Warning: No text extracted from page {idx+1}")

if __name__ == "__main__":
    extract_text("submission.pdf", "submission_text.txt")
