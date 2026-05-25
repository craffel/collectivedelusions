import pypdf

def extract_key_sections(pdf_path):
    print(f"==================== {pdf_path} ====================")
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    
    # Read first 2 pages
    first_two = ""
    for idx in range(min(2, num_pages)):
        first_two += f"--- Page {idx+1} ---\n" + reader.pages[idx].extract_text() + "\n"
        
    # Read last 2 pages
    last_two = ""
    for idx in range(max(0, num_pages - 2), num_pages):
        last_two += f"--- Page {idx+1} ---\n" + reader.pages[idx].extract_text() + "\n"
        
    print("--- FIRST TWO PAGES ---")
    print(first_two[:4000])
    print("--- LAST TWO PAGES ---")
    print(last_two[:4000])
    print("========================================\n\n")

for i in range(3):
    extract_key_sections(f"papers/{i}.pdf")
