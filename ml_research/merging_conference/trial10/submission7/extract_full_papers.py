import pypdf
import os

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".pdf"):
        filepath = os.path.join(papers_dir, filename)
        txt_path = os.path.join(papers_dir, filename.replace(".pdf", ".txt"))
        print(f"Extracting {filename} to {txt_path}...")
        try:
            reader = pypdf.PdfReader(filepath)
            num_pages = len(reader.pages)
            text_by_page = []
            for i in range(num_pages):
                page_text = reader.pages[i].extract_text()
                text_by_page.append(f"--- PAGE {i+1} ---\n{page_text}")
            full_text = "\n\n".join(text_by_page)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"Extracted {num_pages} pages successfully.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
