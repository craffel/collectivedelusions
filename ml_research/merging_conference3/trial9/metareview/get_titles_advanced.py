import glob
import re

for i in range(1, 11):
    files = sorted(glob.glob(f"submission{i}/reviewer*/review.md"))
    if not files:
        print(f"Submission {i}: No reviews found")
        continue
    
    titles_found = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Search for "Paper Title:" or "# Peer Review: " or similar
        m = re.search(r'(?:Paper Title|Title)\s*:?\s*\*?\*?([^\n\*]+)\*?\*?', content, re.IGNORECASE)
        if m:
            titles_found.append(m.group(1).strip())
            
        # Also look at headers starting with #
        first_header = re.search(r'^#\s*(.*)', content, re.MULTILINE)
        if first_header:
            h = first_header.group(1).strip()
            h = re.sub(r'^(?:Peer Review Report: |Peer Review: Report: |Peer Review: |Review Report: |Peer Review Report on |Peer Review Report for |Peer Review of Conference Submission: |Peer Review for Conference Submission: |Peer Review of |Comprehensive Peer Review of |Comprehensive Peer Review: )', '', h, flags=re.IGNORECASE)
            h = re.sub(r'^Peer Review:\s*', '', h, flags=re.IGNORECASE)
            titles_found.append(h)
            
    # Print unique found titles that look plausible
    plausible = []
    for t in titles_found:
        t_clean = t.replace('**', '').replace('__', '').strip()
        if len(t_clean) > 10 and not any(word in t_clean.lower() for word in ["peer review", "conference submission", "review report", "comprehensive peer review", "final peer review"]):
            plausible.append(t_clean)
            
    print(f"Submission {i} plausible titles: {set(plausible)}")
