import pypdf
import os

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".pdf"):
        filepath = os.path.join(papers_dir, filename)
        print(f"=== Reading {filepath} ===")
        try:
            reader = pypdf.PdfReader(filepath)
            num_pages = len(reader.pages)
            print(f"Number of pages: {num_pages}")
            
            # Extract first page
            print("--- PAGE 1 ---")
            print(reader.pages[0].extract_text()[:4000])
            
            # Extract second page
            if num_pages > 1:
                print("--- PAGE 2 ---")
                print(reader.pages[1].extract_text()[:4000])
                
            # Extract last page
            if num_pages > 2:
                print(f"--- LAST PAGE (PAGE {num_pages}) ---")
                print(reader.pages[-1].extract_text()[:4000])
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
        print("\n" + "="*50 + "\n")
