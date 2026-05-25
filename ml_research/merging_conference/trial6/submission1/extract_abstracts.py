import pypdf
import os

pdf_files = ["submission2.pdf", "submission7.pdf", "submission8.pdf"]

for pdf_file in pdf_files:
    pdf_path = os.path.join("papers", pdf_file)
    txt_path = os.path.join("papers", pdf_file.replace(".pdf", ".txt"))
    
    print(f"--- Extracting {pdf_file} ---")
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"Successfully extracted {len(reader.pages)} pages to {txt_path}.")
        print("First 1000 characters:")
        print(text[:1000])
        print("="*60)
    except Exception as e:
        print(f"Error extracting {pdf_file}: {e}")
