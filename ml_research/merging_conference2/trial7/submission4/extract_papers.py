import pypdf
import os

papers_dir = "papers"
for filename in os.listdir(papers_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(papers_dir, filename)
        txt_path = os.path.join(papers_dir, filename.replace(".pdf", ".txt"))
        print(f"Extracting {pdf_path} -> {txt_path}")
        try:
            reader = pypdf.PdfReader(pdf_path)
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"--- Page {i+1} ---\n" + page_text + "\n"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Successfully extracted {len(reader.pages)} pages.")
        except Exception as e:
            print(f"Error extracting {pdf_path}: {e}")
