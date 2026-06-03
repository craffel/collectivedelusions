import pypdf

def extract_intro(pdf_path):
    print(f"=== {pdf_path} ===")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    # Extract first 3 pages
    for i in range(min(4, len(reader.pages))):
        text += reader.pages[i].extract_text()
    
    # Print the first 3000 characters
    print(text[:4000])
    print("\n" + "="*40 + "\n")

extract_intro("papers/submission1.pdf")
extract_intro("papers/submission5.pdf")
extract_intro("papers/submission9.pdf")
