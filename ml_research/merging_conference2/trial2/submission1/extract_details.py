import pypdf
import re

def search_text(pdf_path, terms):
    print(f"=== Searching in {pdf_path} ===")
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        full_text += f"\n--- Page {i+1} ---\n" + page.extract_text()
    
    for term in terms:
        print(f"\n--- Matches for term: '{term}' ---")
        matches = [m.start() for m in re.finditer(term, full_text, re.IGNORECASE)]
        for idx in matches[:5]: # show up to 5 matches
            start = max(0, idx - 150)
            end = min(len(full_text), idx + 250)
            print(full_text[start:end])
            print("..." + "-"*15 + "...")

search_text("papers/submission5.pdf", ["hyperparameter", "epoch", "learning rate", "optimizer", "resnet-18", "calibration"])
search_text("papers/submission1.pdf", ["hyperparameter", "epoch", "learning rate", "optimizer", "resnet-18", "calibration"])
