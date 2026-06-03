import pypdf

def extract_pdf_text(pdf_path, output_path):
    reader = pypdf.PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    full_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        full_text.append(f"--- Page {i+1} ---")
        full_text.append(text)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))
    print(f"Text successfully extracted and saved to {output_path}")

if __name__ == "__main__":
    extract_pdf_text("submission.pdf", "submission_text.txt")
