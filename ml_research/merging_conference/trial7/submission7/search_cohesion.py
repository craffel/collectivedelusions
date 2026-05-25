import pypdf

def search_text(pdf_path, word):
    print(f"\n--- Searching {pdf_path} for '{word}' ---")
    reader = pypdf.PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if word.lower() in text.lower():
            for line in text.split('\n'):
                if word.lower() in line.lower() and len(line) > 20:
                    print(f"Page {i+1}: {line[:120]}")

search_text("papers/submission3.pdf", "cohesion")
search_text("papers/submission6.pdf", "cohesion")
