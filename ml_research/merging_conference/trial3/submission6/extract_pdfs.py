import pypdf
import os

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"--- Page {i+1} ---\n" + (page_text or "") + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

os.makedirs("papers_txt", exist_ok=True)
for filename in os.listdir("papers"):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join("papers", filename)
        txt_path = os.path.join("papers_txt", filename.replace(".pdf", ".txt"))
        extract_pdf_to_txt(pdf_path, txt_path)
print("Done!")
