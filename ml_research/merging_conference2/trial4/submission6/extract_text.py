import sys
import pypdf

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = []
    for i, page in enumerate(reader.pages):
        text.append(f"--- Page {i+1} ---")
        text.append(page.extract_text())
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_text.py <pdf_path> <txt_path>")
        sys.exit(1)
    extract_pdf_text(sys.argv[1], sys.argv[2])
