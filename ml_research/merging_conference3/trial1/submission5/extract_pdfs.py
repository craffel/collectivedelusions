import pypdf

def extract_pdf_info(pdf_path, txt_path):
    print(f"Extracting {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    meta = reader.metadata
    print(f"Metadata: {meta}")
    
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"--- PAGE {i} ---\n" + page_text + "\n"
            
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {len(reader.pages)} pages to {txt_path}\n")

extract_pdf_info("papers/0.pdf", "papers/0.txt")
extract_pdf_info("papers/1.pdf", "papers/1.txt")
extract_pdf_info("papers/2.pdf", "papers/2.txt")
