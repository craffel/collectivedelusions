import pypdf

def check_pages(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages in {pdf_path}: {len(reader.pages)}")
    for i in [7, 8, 9]:
        if i < len(reader.pages):
            text = reader.pages[i].extract_text()
            print(f"\n--- PAGE {i+1} START ---")
            print(text[:400])
            print("...")
            print(text[-400:])
            print(f"--- PAGE {i+1} END ---\n")

if __name__ == '__main__':
    check_pages('submission.pdf')
