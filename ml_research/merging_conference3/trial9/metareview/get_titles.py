import glob
import re

for i in range(1, 11):
    files = sorted(glob.glob(f"submission{i}/reviewer*/review.md"))
    if not files:
        print(f"Submission {i}: No reviews found")
        continue
    # Look for title in the first file
    with open(files[0], 'r', encoding='utf-8') as f:
        content = f.read()
    
    title = ""
    # Try looking for "Paper Title:" or "# Peer Review:" or first heading
    pt_match = re.search(r'Paper Title:\s*(.*)', content, re.IGNORECASE)
    if pt_match:
        title = pt_match.group(1).strip()
    else:
        first_line = content.split('\n')[0]
        if first_line.startswith('#'):
            title = first_line.replace('#', '').strip()
            title = re.sub(r'^(?:Peer Review Report: |Peer Review: Report: |Peer Review: |Review Report: |Peer Review Report on |Peer Review Report for |Peer Review of Conference Submission: |Peer Review for Conference Submission: |Peer Review of |Comprehensive Peer Review of |Comprehensive Peer Review: )', '', title, flags=re.IGNORECASE)
            title = re.sub(r'^Peer Review:\s*', '', title, flags=re.IGNORECASE)
            
    print(f"Submission {i} title: {title}")
