import pypdf
import os

pdf_dir = "papers"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    txt_path = pdf_file.replace(".pdf", ".txt")
    print(f"Extracting {pdf_path} to {txt_path}...")
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                f.write(f"\n--- PAGE {i+1} ---\n")
                text = page.extract_text()
                if text:
                    f.write(text)
        print(f"Done. Extracted {len(reader.pages)} pages.")
    except Exception as e:
        print(f"Error extracting {pdf_file}: {e}")
