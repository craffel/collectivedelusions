import pypdf
import os

pdf_dir = "papers"
pdf_files = ["submission3.pdf", "submission6.pdf", "submission10.pdf"]

for f in pdf_files:
    path = os.path.join(pdf_dir, f)
    out_path = f.replace(".pdf", ".txt")
    print(f"Extracting {f} to {out_path}...")
    try:
        reader = pypdf.PdfReader(path)
        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, page in enumerate(reader.pages):
                out_f.write(f"=== Page {i+1} ===\n")
                out_f.write(page.extract_text() or "")
                out_f.write("\n\n")
        print(f"Finished extracting {f} to {out_path}.")
    except Exception as e:
        print(f"Error: {e}")
