import pypdf

def inspect_pdf(pdf_path):
    print(f"=== Inspecting {pdf_path} ===")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    # Extract first page
    first_page = reader.pages[0]
    text = first_page.extract_text()
    print("--- FIRST PAGE TEXT ---")
    print(text[:2000])  # limit to first 2000 chars
    print("-----------------------\n")

for i in range(3):
    inspect_pdf(f"papers/{i}.pdf")
