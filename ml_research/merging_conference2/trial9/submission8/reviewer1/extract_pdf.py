import pypdf

def extract_text(pdf_path, txt_path):
    print(f"Reading {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            f.write(f"--- PAGE {i+1} ---\n")
            text = page.extract_text()
            if text:
                f.write(text)
                f.write("\n")
            else:
                f.write("[No text found on this page]\n")
    print(f"Done! Text saved to {txt_path}")

if __name__ == "__main__":
    extract_text("submission.pdf", "submission_text.txt")
