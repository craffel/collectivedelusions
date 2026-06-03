import pypdf
import os

papers = ["submission3.pdf", "submission7.pdf", "submission9.pdf"]
for paper in papers:
    pdf_path = os.path.join("papers", paper)
    txt_path = os.path.join("papers", paper.replace(".pdf", ".txt"))
    print(f"Extracting {pdf_path} to {txt_path}...")
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} does not exist.")
        continue
        
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(f"--- PAGE {i+1} ---")
                text.append(page_text)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text))
        print(f"Successfully extracted {len(reader.pages)} pages.")
    except Exception as e:
        print(f"Failed to extract {pdf_path}: {e}")
