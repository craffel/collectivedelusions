import os
import re

base_dir = "/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial5/metareview"

def extract_title(content):
    # Try parsing first line
    lines = content.split('\n')
    first_line = lines[0].strip()
    if first_line.startswith("# Peer Review:"):
        return first_line.replace("# Peer Review:", "").strip()
    elif first_line.startswith("# Peer Review"):
        # Look for title in the next few lines or headers
        for line in lines[1:10]:
            if line.startswith("# "):
                return line.replace("# ", "").strip()
            if "Title:" in line:
                return line.split("Title:", 1)[1].strip()
    # Fallback: find any line starting with # Peer Review
    for line in lines:
        if line.startswith("# Peer Review:"):
            return line.replace("# Peer Review:", "").strip()
    return "Unknown Title"

def extract_score(content, file_path):
    # Look for overall recommendation section or scores
    # Search for lines containing Score:, Rating:, Overall Recommendation, etc.
    lines = content.split('\n')
    score_line = None
    
    # Let's search from the bottom of the file or find section headers
    for i, line in enumerate(lines):
        if any(term in line.lower() for term in ["overall recommendation", "rating:", "score:"]):
            # Check if this line or the next few lines have a score (like 1-6)
            for offset in range(0, 4):
                if i + offset < len(lines):
                    l = lines[i + offset].strip()
                    # Look for digits or standard recommendation phrasing
                    if re.search(r'\b[1-6]\b', l) or "accept" in l.lower() or "reject" in l.lower():
                        return l
    
    # Try regex search
    match = re.search(r'(?:Score|Rating|Recommendation)\s*:\s*(.*)', content, re.IGNORECASE)
    if match:
        return match.group(1).strip()
        
    return "Not Found"

def extract_strengths_weaknesses(content):
    # Locate Strengths and Weaknesses section
    # Let's search for headers like "## Strengths and Weaknesses", "### Strengths", "### Weaknesses"
    lines = content.split('\n')
    strengths = []
    weaknesses = []
    
    current_section = None
    for line in lines:
        l_lower = line.lower().strip()
        if "### strengths" in l_lower or "#### strengths" in l_lower or "1. **strengths**" in l_lower or "1. **main strengths" in l_lower or "### main strengths" in l_lower:
            current_section = "strengths"
            continue
        elif "### weaknesses" in l_lower or "#### weaknesses" in l_lower or "2. **weaknesses**" in l_lower or "3. **major weaknesses" in l_lower or "### major weaknesses" in l_lower:
            current_section = "weaknesses"
            continue
        elif line.startswith("## ") or (line.startswith("---") and current_section):
            # End current section if a new main section starts
            if not any(term in l_lower for term in ["strength", "weakness"]):
                current_section = None
                
        if current_section == "strengths":
            if line.strip() and not line.startswith("###") and not line.startswith("##"):
                strengths.append(line.strip())
        elif current_section == "weaknesses":
            if line.strip() and not line.startswith("###") and not line.startswith("##"):
                weaknesses.append(line.strip())
                
    # Format them nicely (first 3-4 bullet points or lines)
    str_summary = "\n".join(strengths[:8])
    weak_summary = "\n".join(weaknesses[:8])
    return str_summary, weak_summary

results = []

for n in range(1, 11):
    sub_dir = os.path.join(base_dir, f"submission{n}")
    if not os.path.isdir(sub_dir):
        continue
    
    sub_data = {
        "num": n,
        "title": "Unknown",
        "reviews": []
    }
    
    titles_found = []
    for r in range(1, 4):
        rev_path = os.path.join(sub_dir, f"reviewer{r}", "review.md")
        if os.path.exists(rev_path):
            with open(rev_path, 'r', encoding='utf-8') as f:
                content = f.read()
            title = extract_title(content)
            if title and title != "Unknown Title":
                titles_found.append(title)
            score = extract_score(content, rev_path)
            str_sum, weak_sum = extract_strengths_weaknesses(content)
            
            sub_data["reviews"].append({
                "reviewer": r,
                "score": score,
                "strengths": str_sum,
                "weaknesses": weak_sum,
                "full_content": content
            })
            
    if titles_found:
        # Use the most common or first title found
        sub_data["title"] = max(set(titles_found), key=titles_found.count)
    results.append(sub_data)

# Print a nice summary report
print("================================================================================")
print("SUBMISSIONS SUMMARY")
print("================================================================================")
for sub in results:
    print(f"\nSubmission {sub['num']}: {sub['title']}")
    for rev in sub["reviews"]:
        print(f"  Reviewer {rev['reviewer']}: {rev['score']}")
