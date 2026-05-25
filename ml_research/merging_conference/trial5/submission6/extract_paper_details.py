import pypdf
import os

pdf_dir = "papers"
for pdf_file in sorted(os.listdir(pdf_dir)):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        txt_path = pdf_path.replace(".pdf", ".txt")
        print(f"Extracting {pdf_path} -> {txt_path} ...")
        try:
            reader = pypdf.PdfReader(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                for page_idx, page in enumerate(reader.pages):
                    f.write(f"--- Page {page_idx + 1} ---\n")
                    f.write(page.extract_text() or "")
                    f.write("\n\n")
            print(f"Finished extracting {pdf_file}")
        except Exception as e:
            print(f"Error extracting {pdf_file}: {e}")
