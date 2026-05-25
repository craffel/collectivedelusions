import pypdf
import os

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".pdf"):
        filepath = os.path.join(papers_dir, filename)
        print(f"=== {filename} ===")
        try:
            reader = pypdf.PdfReader(filepath)
            num_pages = len(reader.pages)
            print(f"Number of pages: {num_pages}")
            # Extract first 2 pages
            text = ""
            for i in range(min(2, num_pages)):
                text += reader.pages[i].extract_text()
            print(text[:2000]) # Print first 2000 characters
            print("\n" + "="*40 + "\n")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
