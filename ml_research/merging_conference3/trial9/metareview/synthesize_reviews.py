import glob
import os
import re

files = sorted(glob.glob("submission*/reviewer*/review.md"))

submissions_data = {}

for file_path in files:
    parts = file_path.split('/')
    sub_num = int(parts[0].replace('submission', ''))
    rev_num = int(parts[1].replace('reviewer', ''))
    
    if sub_num not in submissions_data:
        submissions_data[sub_num] = {
            "title": "",
            "reviewers": {},
        }
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    title_m = re.search(r'^#\s*(.*)', content)
    title = title_m.group(1).strip() if title_m else ""
    # clean up potential markdown title headers
    title = re.sub(r'^(?:Peer Review: Report on |Peer Review Report: |Peer Review: Report: |Peer Review: |Review Report: )', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^(?:Peer Review Report on |Peer Review: )', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^Peer Review:\s*', '', title, flags=re.IGNORECASE)
    
    if title and not submissions_data[sub_num]["title"]:
        submissions_data[sub_num]["title"] = title
        
    # Find rating
    rating = "None"
    # Search for recommendation/rating in markdown
    for line in content.split('\n'):
        line_l = line.lower()
        if "rating:" in line_l or "score:" in line_l or "recommendation:" in line_l or "overall rating:" in line_l:
            cleaned = line.replace('**', '').replace('__', '').strip()
            # extract the actual rating value (e.g., "5: Accept", "3: Weak reject")
            m = re.search(r'(?:Rating|Score|Recommendation|Overall Rating)\s*:?\s*(.*)', cleaned, re.IGNORECASE)
            if m:
                rating = m.group(1).strip()
                break
                
    # Extract Strengths and Weaknesses
    strengths_section = ""
    str_m = re.search(r'###\s*(?:Major\s*)?Strengths\s*:?\s*\n*(.*?)\n*(?:###|##|$)', content, re.DOTALL | re.IGNORECASE)
    if not str_m:
        str_m = re.search(r'Strengths\s*\n*(.*?)\n*(?:Weaknesses|##|$)', content, re.DOTALL | re.IGNORECASE)
    if str_m:
        strengths_section = str_m.group(1).strip()
        
    weaknesses_section = ""
    weak_m = re.search(r'###\s*(?:Major\s*)?Weaknesses\s*:?\s*\n*(.*?)\n*(?:###|##|$)', content, re.DOTALL | re.IGNORECASE)
    if not weak_m:
        weak_m = re.search(r'Weaknesses\s*\n*(.*?)\n*(?:##|$)', content, re.DOTALL | re.IGNORECASE)
    if weak_m:
        weaknesses_section = weak_m.group(1).strip()
        
    # Extract soundess/presentation/significance/originality ratings if any
    dimension_ratings = {}
    dim_matches = re.findall(r'(\w+)\s*:\s*(?:Rating\s*:?\s*)?\*?\*?(Excellent|Good|Fair|Poor|Very Good)\*?\*?', content, re.IGNORECASE)
    for dim, r in dim_matches:
        dimension_ratings[dim.lower()] = r
        
    submissions_data[sub_num]["reviewers"][rev_num] = {
        "rating": rating,
        "strengths": strengths_section,
        "weaknesses": weaknesses_section,
        "dimensions": dimension_ratings,
        "raw_text_snippet": content[:800]
    }

# Ensure title is set if not found in first line
for sub_num in sorted(submissions_data.keys()):
    if not submissions_data[sub_num]["title"]:
        # Try to infer from files
        submissions_data[sub_num]["title"] = f"Submission {sub_num} Title Placeholder"

# Print synthesized report
print("=== AGGREGATED SUBMISSIONS ANALYSIS ===")
for sub_num in sorted(submissions_data.keys()):
    sub = submissions_data[sub_num]
    print(f"\n==================================================")
    print(f"SUBMISSION {sub_num}: {sub['title']}")
    print(f"Reviewers active: {list(sub['reviewers'].keys())}")
    for rev_num, r_data in sub["reviewers"].items():
        print(f"  - Reviewer {rev_num}: Rating = {r_data['rating']}")
        # print first few strengths/weaknesses bullet points or lines
        str_lines = [l.strip() for l in r_data['strengths'].split('\n') if l.strip()][:3]
        weak_lines = [l.strip() for l in r_data['weaknesses'].split('\n') if l.strip()][:3]
        print(f"    Strengths (top 3): {str_lines}")
        print(f"    Weaknesses (top 3): {weak_lines}")
    print(f"==================================================")
