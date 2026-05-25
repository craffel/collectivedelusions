import pypdf

def extract_detailed_text(pdf_path, keywords, output_file):
    reader = pypdf.PdfReader(pdf_path)
    print(f"Reading {pdf_path}")
    
    with open(output_file, "w") as f:
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            lines = text.split('\n')
            
            # If any keyword is in the page, write the page's text to file
            if any(kw.lower() in text.lower() for kw in keywords):
                f.write(f"\n--- PAGE {idx+1} ---\n")
                f.write(text)
                print(f"Page {idx+1} matched keywords.")

extract_detailed_text("papers/submission1.pdf", ["methodology", "sbf", "rgp", "formula", "equation"], "sata_details.txt")
