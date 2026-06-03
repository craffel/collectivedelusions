import pypdf
import os

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        text += f"--- PAGE {page_num+1} ---\n"
        text += page.extract_text() + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    for i in range(3):
        pdf_path = f"papers/{i}.pdf"
        txt_path = f"papers/{i}.txt"
        if os.path.exists(pdf_path):
            extract_pdf_to_txt(pdf_path, txt_path)
        else:
            print(f"{pdf_path} does not exist.")
