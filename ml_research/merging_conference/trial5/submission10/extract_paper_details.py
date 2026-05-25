import pypdf

def extract_pages(pdf_path, start_page, end_page):
    print(f"=== {pdf_path} (Pages {start_page}-{end_page}) ===")
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    
    text = ""
    for i in range(start_page-1, min(end_page, num_pages)):
        text += f"--- Page {i+1} ---\n"
        text += reader.pages[i].extract_text() or ""
    print(text[:8000]) # print up to 8000 characters
    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    extract_pages("papers/submission6.pdf", 3, 7)
    extract_pages("papers/submission8.pdf", 3, 7)
    extract_pages("papers/submission9.pdf", 3, 7)
