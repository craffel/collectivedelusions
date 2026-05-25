import re

def search_sections(txt_path):
    print(f"\n==================== {txt_path} ====================")
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Let's find sections 3 and 4
    # Usually sections are numbered like "3. Methodology" or "3. Proposed Method" or "4. Experiments"
    pattern = re.compile(r'(?:^|\n)(3\.\s+\w+.*?)(?=\n[45]\.\s+|$)', re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(content)
    for m in matches:
        print("--- Section 3 ---")
        print(m[:3000])
        
    pattern2 = re.compile(r'(?:^|\n)(4\.\s+\w+.*?)(?=\n[56]\.\s+|$)', re.DOTALL | re.IGNORECASE)
    matches2 = pattern2.findall(content)
    for m in matches2:
        print("--- Section 4 ---")
        print(m[:3000])

import os
for paper in sorted(os.listdir("papers_txt")):
    if paper.endswith(".txt"):
        search_sections(os.path.join("papers_txt", paper))
