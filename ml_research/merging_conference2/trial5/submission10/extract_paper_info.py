import pypdf

def extract_abstract(pdf_path):
    print(f"=== Extracting from: {pdf_path} ===")
    reader = pypdf.PdfReader(pdf_path)
    # Extract first 2 pages (usually contains title, abstract, and introduction)
    text = ""
    for idx in range(min(len(reader.pages), 2)):
        text += f"\n--- Page {idx+1} ---\n"
        text += reader.pages[idx].extract_text()
    print(text[:3000]) # Limit length
    print("\n" + "="*40 + "\n")

extract_abstract("papers/submission3.pdf")
extract_abstract("papers/submission9.pdf")
extract_abstract("papers/submission10.pdf")
