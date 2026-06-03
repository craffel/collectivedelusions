import pypdf

def search_pdf(pdf_path, word):
    reader = pypdf.PdfReader(pdf_path)
    print(f"Searching '{word}' in {pdf_path}:")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if word.lower() in text.lower():
            print(f"Found on page {i+1}")
            # print surrounding text of the first match
            idx = text.lower().find(word.lower())
            start = max(0, idx - 200)
            end = min(len(text), idx + 300)
            print(f"... {text[start:end]} ...")
            print("-" * 40)

if __name__ == '__main__':
    search_pdf("papers/submission3.pdf", "task")
    search_pdf("papers/submission3.pdf", "conditional")
