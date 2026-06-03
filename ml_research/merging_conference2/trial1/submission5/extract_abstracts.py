import pypdf
import os

def extract_first_page_text(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        if len(reader.pages) > 0:
            return reader.pages[0].extract_text()
        return "Empty PDF"
    except Exception as e:
        return f"Error reading PDF: {e}"

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".pdf"):
        path = os.path.join(papers_dir, filename)
        print(f"=== {filename} ===")
        text = extract_first_page_text(path)
        # print first 1500 characters
        print(text[:1500])
        print("\n" + "="*40 + "\n")
