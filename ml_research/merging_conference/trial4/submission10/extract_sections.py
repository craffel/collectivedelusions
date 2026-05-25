import os
from pypdf import PdfReader

papers_dir = "papers"
papers = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])

for paper in papers:
    path = os.path.join(papers_dir, paper)
    print(f"\n==================================================")
    print(f"Paper: {paper}")
    print(f"==================================================")
    try:
        reader = PdfReader(path)
        print(f"Total pages: {len(reader.pages)}")
        
        # We can extract all headers or search for "Method", "Experiment", "Result" sections
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            lines = text.split("\n")
            for line in lines:
                # Check for headings (e.g., "1. ", "2. ", "3. ", "4. ", "5. ", "6. ")
                line_stripped = line.strip()
                if any(line_stripped.startswith(f"{i}. ") for i in range(1, 10)) or any(line_stripped.startswith(f"{i}.{j} ") for i in range(1, 10) for j in range(1, 10)):
                    print(f"Page {idx+1}: {line_stripped}")
                    
    except Exception as e:
        print(f"Error: {e}")
