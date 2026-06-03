import pypdf

def extract_first_page(pdf_path):
    print(f"\n================== {pdf_path} ==================")
    reader = pypdf.PdfReader(pdf_path)
    if len(reader.pages) > 0:
        first_page = reader.pages[0]
        text = first_page.extract_text()
        print(text[:1500])  # Print first 1500 characters
    else:
        print("Empty PDF")

if __name__ == "__main__":
    extract_first_page("papers/submission3.pdf")
    extract_first_page("papers/submission7.pdf")
    extract_first_page("papers/submission8.pdf")
