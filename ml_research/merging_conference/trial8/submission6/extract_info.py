import pypdf
import os

pdf_dir = "papers"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    print(f"=== {pdf_file} ===")
    try:
        reader = pypdf.PdfReader(pdf_path)
        print(f"Number of pages: {len(reader.pages)}")
        # Print first page or two to find title/abstract
        text = ""
        for i in range(min(2, len(reader.pages))):
            text += reader.pages[i].extract_text() + "\n"
        print(text[:2500]) # Print first 2500 chars
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
    print("\n" + "="*50 + "\n")
