import pypdf
import os

for filename in sorted(os.listdir('papers')):
    if filename.endswith('.pdf'):
        path = os.path.join('papers', filename)
        print(f"=== {filename} ===")
        try:
            reader = pypdf.PdfReader(path)
            print(f"Pages: {len(reader.pages)}")
            # Print first 2 pages (or up to total pages)
            for i in range(min(2, len(reader.pages))):
                print(f"--- Page {i+1} ---")
                text = reader.pages[i].extract_text()
                print(text[:1500])  # Print first 1500 chars of each page
        except Exception as e:
            print(f"Error reading {filename}: {e}")
        print("\n" + "="*40 + "\n")
