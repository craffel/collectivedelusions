import re
import os

papers_dir = "papers"
for filename in sorted(os.listdir(papers_dir)):
    if filename.endswith(".txt"):
        filepath = os.path.join(papers_dir, filename)
        print(f"=== STRUCTURE OF {filename} ===")
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Simple heuristic for section headings (e.g., "1. Introduction", "2. Related Work")
        heading_re = re.compile(r"^(?:\d+\.|\d+\.\d+)\s+[A-Z][A-Za-z\s\-]+$")
        
        headings_found = []
        for i, line in enumerate(lines):
            line_str = line.strip()
            if heading_re.match(line_str):
                headings_found.append((i+1, line_str))
                
        for line_num, heading in headings_found[:15]:
            print(f"Line {line_num:4d}: {heading}")
        
        # Also print abstract
        print("\n--- ABSTRACT SEARCH ---")
        abstract_start = -1
        intro_start = -1
        for i, line in enumerate(lines):
            if "Abstract" in line:
                abstract_start = i
            if "1. Introduction" in line or "1 Introduction" in line:
                intro_start = i
                break
        if abstract_start != -1 and intro_start != -1:
            print("".join(lines[abstract_start:intro_start]))
        else:
            print("Could not find abstract or introduction cleanly.")
            
        print("\n" + "="*50 + "\n")
