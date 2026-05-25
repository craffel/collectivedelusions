import pypdf

def extract_pdf_info(pdf_path):
    print(f"\n--- Extracting {pdf_path} ---")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    
    # Extract first page
    first_page_text = reader.pages[0].extract_text()
    print("First Page Content (or part of it):")
    print(first_page_text[:1500])
    print("\n------------------------")
    
    # Let's search for some keywords like "Fisher", "EBER", "precondition", "merging", "Riemannian"
    print("Keyword search:")
    keywords = ["Fisher", "EBER", "precondition", "Riemannian", "gradient", "entropy"]
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        for kw in keywords:
            if kw.lower() in text.lower():
                # print a line containing the keyword
                for line in text.split('\n'):
                    if kw.lower() in line.lower() and len(line) > 10:
                        print(f"Page {i+1} [{kw}]: {line[:120]}")
                        break

print("Extracting Submission 3:")
extract_pdf_info("papers/submission3.pdf")

print("\n\nExtracting Submission 6:")
extract_pdf_info("papers/submission6.pdf")

print("\n\nExtracting Submission 10:")
extract_pdf_info("papers/submission10.pdf")
