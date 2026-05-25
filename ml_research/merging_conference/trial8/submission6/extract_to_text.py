import pypdf
import os

pdf_dir = "papers"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    txt_path = pdf_file.replace(".pdf", "_text.txt")
    print(f"Extracting {pdf_file} to {txt_path}...")
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                f.write(f"--- PAGE {i+1} ---\n")
                f.write(page.extract_text() or "")
                f.write("\n\n")
        print(f"Successfully wrote {txt_path}")
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
