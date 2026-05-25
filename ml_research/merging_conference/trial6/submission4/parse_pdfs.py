import pypdf
import os

pdf_files = ["papers/submission2.pdf", "papers/submission7.pdf", "papers/submission8.pdf"]

for pdf_file in pdf_files:
    txt_file = pdf_file.replace(".pdf", ".txt")
    print(f"Parsing {pdf_file} -> {txt_file}...")
    try:
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"\n--- Page {i+1} ---\n"
            text += page.extract_text()
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully wrote {len(reader.pages)} pages to {txt_file}.")
    except Exception as e:
        print(f"Error parsing {pdf_file}: {e}")
