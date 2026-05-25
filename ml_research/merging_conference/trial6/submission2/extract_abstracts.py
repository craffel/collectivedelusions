import pypdf
import os

def extract_pdf_info(pdf_path, txt_path):
    print(f"Extracting {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text_content = []
    
    # Extract first 3 pages first (usually contains Abstract, Intro, and background)
    num_pages = len(reader.pages)
    text_content.append(f"--- PATH: {pdf_path} (Total Pages: {num_pages}) ---")
    
    # Let's extract all pages but we'll save it to a text file, and we can also print the first page or two here.
    for i in range(num_pages):
        page_text = reader.pages[i].extract_text()
        text_content.append(f"=== PAGE {i+1} ===")
        text_content.append(page_text)
        
    full_text = "\n".join(text_content)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # Print the first 2 pages for quick preview
    print(f"--- Preview of {pdf_path} (First 2 Pages) ---")
    preview = "\n".join(text_content[:4]) # title + page 1 + page 2
    print(preview[:3000])
    print("\n" + "="*50 + "\n")

os.makedirs("extracted_papers", exist_ok=True)
for name in ["submission2.pdf", "submission7.pdf", "submission8.pdf"]:
    extract_pdf_info(os.path.join("papers", name), os.path.join("extracted_papers", name.replace(".pdf", ".txt")))
print("Done extracting all papers!")
