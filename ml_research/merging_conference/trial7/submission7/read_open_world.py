import pypdf

def print_pages(pdf_path, pages):
    reader = pypdf.PdfReader(pdf_path)
    for p in pages:
        if p < len(reader.pages):
            print(f"\n================ {pdf_path} PAGE {p+1} ================")
            print(reader.pages[p].extract_text()[:4000])

print_pages("papers/submission3.pdf", [3, 4, 5])
print_pages("papers/submission6.pdf", [3, 4, 5])
