import os
import pypdf

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text_content = []
    for i, page in enumerate(reader.pages):
        text_content.append(f"--- Page {i+1} ---")
        text_content.append(page.extract_text() or "")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_content))

os.makedirs("papers_text", exist_ok=True)
for filename in os.listdir("papers"):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join("papers", filename)
        txt_path = os.path.join("papers_text", filename.replace(".pdf", ".txt"))
        extract_pdf_to_txt(pdf_path, txt_path)
print("Extraction complete!")
