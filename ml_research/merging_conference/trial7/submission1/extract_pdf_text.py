import pypdf

def extract_pdf(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- PAGE {i+1} ---\n"
        text += page.extract_text() + "\n"
    with open(txt_path, "w") as f:
        f.write(text)
    print("Done.")

extract_pdf("papers/submission3.pdf", "papers/submission3.txt")
extract_pdf("papers/submission6.pdf", "papers/submission6.txt")
extract_pdf("papers/submission10.pdf", "papers/submission10.txt")
