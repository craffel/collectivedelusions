import pypdf

def extract_info(pdf_path):
    print(f"\n==========================================")
    print(f"Extracting info from: {pdf_path}")
    print(f"==========================================")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    
    # Extract page 1 (contains Abstract, Intro)
    text_p1 = reader.pages[0].extract_text()
    print("--- PAGE 1 ---")
    print(text_p1[:1500]) # first 1500 chars
    
    # Let's search for "Method" or "Proposed" in pages
    print("--- SECTION SEARCH ---")
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        for line in text.split('\n'):
            line_strip = line.strip().lower()
            if any(h in line_strip for h in ["abstract", "introduction", "method", "proposed method", "experiments", "results"]):
                print(f"[Page {idx+1}] {line.strip()[:100]}")

extract_info("papers/submission1.pdf")
extract_info("papers/submission5.pdf")
extract_info("papers/submission7.pdf")
