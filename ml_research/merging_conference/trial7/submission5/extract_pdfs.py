import os
try:
    import pypdf
except ImportError:
    print("pypdf not found, please install or run with uv run --with pypdf")

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- PAGE {i+1} ---\n"
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(reader.pages)} pages.")

if __name__ == "__main__":
    os.makedirs("extracted_papers", exist_ok=True)
    for name in ["submission3", "submission6", "submission10"]:
        pdf = f"papers/{name}.pdf"
        txt = f"extracted_papers/{name}.txt"
        if os.path.exists(pdf):
            extract_pdf_text(pdf, txt)
        else:
            print(f"File {pdf} not found.")
