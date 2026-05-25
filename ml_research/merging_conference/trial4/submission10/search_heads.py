from pypdf import PdfReader

def search_text(file_path):
    print(f"\nSearching in {file_path}:")
    reader = PdfReader(file_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if "active head" in text.lower() or "active classification" in text.lower() or "active task" in text.lower():
            print(f"--- Page {i+1} ---")
            for line in text.split("\n"):
                if any(k in line.lower() for k in ["active head", "active classification", "active task", "predict", "head"]):
                    print(line)

search_text("papers/submission1.pdf")
search_text("papers/submission7.pdf")
