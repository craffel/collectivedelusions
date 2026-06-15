import os
import re

base_dir = "/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial5/metareview"

def get_clean_score(text):
    # Regex to find scores in text, typically like "6: Strong Accept", "6", "Score: 6", "Rating: 6"
    # Search for patterns like rating: 6, score: 6, rating: 6, 6: strong accept, 2 (reject), score: 2
    patterns = [
        r"(?:score|rating|recommendation)\s*[:\*]*\s*([1-6])\s*[:\-\(]*\s*([A-Za-z ]+)?",
        r"(?:score|rating|recommendation)\s*[:\*]*\s*\*?\*?([1-6])\s*[:\-\(]*\s*([A-Za-z ]+)?",
        r"\b([1-6])\s*:\s*(?:Strong Accept|Accept|Weak Accept|Weak Reject|Reject|Strong Reject)\b",
        r"rating\s*:\s*([1-6])",
        r"score\s*:\s*([1-6])",
    ]
    
    # Let's search line by line first, focusing on lines with "recommendation", "rating", "score"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if any(term in line.lower() for term in ["recommendation", "score", "rating", "justification"]):
            # Check this and next 3 lines
            for offset in range(0, 4):
                if i + offset < len(lines):
                    l = lines[i + offset].strip()
                    # Look for something like "6: Strong Accept" or "**3: Weak Reject**"
                    m = re.search(r'\b([1-6])\s*:\s*([A-Za-z ]+)', l)
                    if m:
                        return f"{m.group(1)} ({m.group(2).strip()})"
                    # Look for "**Score: 2 (Reject)**"
                    m = re.search(r'score\s*:\s*([1-6])\s*\(?([A-Za-z ]+)?\)?', l, re.IGNORECASE)
                    if m:
                        desc = m.group(2).strip() if m.group(2) else ""
                        return f"{m.group(1)} ({desc})" if desc else m.group(1)
                    # Look for "Rating: 4"
                    m = re.search(r'rating\s*:\s*([1-6])\s*\(?([A-Za-z ]+)?\)?', l, re.IGNORECASE)
                    if m:
                        desc = m.group(2).strip() if m.group(2) else ""
                        return f"{m.group(1)} ({desc})" if desc else m.group(1)
                    # Look for "* **Rating:** 4 (Weak Accept)"
                    m = re.search(r'rating\s*:\s*\*?\*?\s*([1-6])\s*\(?([A-Za-z ]+)?\)?', l, re.IGNORECASE)
                    if m:
                        desc = m.group(2).strip() if m.group(2) else ""
                        return f"{m.group(1)} ({desc})" if desc else m.group(1)
                    # Look for "**Recommendation: 2: Reject**"
                    m = re.search(r'recommendation\s*:\s*([1-6])\s*[:\-]\s*([A-Za-z ]+)', l, re.IGNORECASE)
                    if m:
                        return f"{m.group(1)} ({m.group(2).strip()})"
                    # Look for "Score: 6: Strong Accept"
                    m = re.search(r'score\s*:\s*([1-6])\s*:\s*([A-Za-z ]+)', l, re.IGNORECASE)
                    if m:
                        return f"{m.group(1)} ({m.group(2).strip()})"

    # Fallback to search of the entire text
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = m.group(1)
            desc = m.group(2).strip() if len(m.groups()) > 1 and m.group(2) else ""
            return f"{val} ({desc})" if desc else val
            
    # Hardcoded fallbacks if we can find specific lines near the end
    # Let's search the last 20 lines for any line containing a digit
    for line in reversed(lines[-25:]):
        m = re.search(r'\b([1-6])\b', line)
        if m and any(term in line.lower() for term in ["accept", "reject", "score", "rating", "recommendation"]):
            return line.strip()
            
    return "Not Found"

def extract_title(content):
    lines = content.split('\n')
    for line in lines[:10]:
        if line.startswith("# Peer Review:"):
            return line.replace("# Peer Review:", "").strip()
        elif line.startswith("# "):
            # strip "Peer Review:" or similar
            title = line.replace("# ", "").strip()
            if "peer review" in title.lower():
                title = re.sub(r'peer review\s*:\s*', '', title, flags=re.IGNORECASE)
                title = re.sub(r'peer review\s*', '', title, flags=re.IGNORECASE)
            return title
    return "Unknown Title"

results = []

for n in range(1, 11):
    sub_dir = os.path.join(base_dir, f"submission{n}")
    if not os.path.isdir(sub_dir):
        continue
    
    title = "Unknown"
    scores = []
    
    for r in range(1, 4):
        filepath = os.path.join(sub_dir, f"reviewer{r}", "review.md")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if title == "Unknown":
                title = extract_title(content)
            score = get_clean_score(content)
            scores.append((r, score))
            
    results.append({
        "num": n,
        "title": title,
        "scores": scores
    })

for r in results:
    print(f"Submission {r['num']}: {r['title']}")
    for reviewer, score in r['scores']:
        print(f"  Reviewer {reviewer}: {score}")
