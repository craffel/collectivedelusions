import pypdf

def extract_metadata_and_text(pdf_path):
    print(f"=== Reading {pdf_path} ===")
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"Number of pages: {num_pages}")
    
    # Print first page or two to get title/abstract/intro
    text = ""
    for i in range(min(2, num_pages)):
        text += f"--- Page {i+1} ---\n"
        text += reader.pages[i].extract_text() or ""
    print(text[:4000]) # Print first 4000 chars of extracted text
    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    for paper in ["papers/submission6.pdf", "papers/submission8.pdf", "papers/submission9.pdf"]:
        extract_metadata_and_text(paper)
