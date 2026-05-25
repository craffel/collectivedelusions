import re

files = ["submission3.txt", "submission6.txt", "submission10.txt"]

def extract_section(text, sec_name):
    # Try to find a section starting with sec_name
    pattern = r'(?i)^\s*(?:\d+\.?\s+)?' + re.escape(sec_name) + r'.*?$'
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return "Section not found."
    start_idx = match.start()
    # Find the next section (a line starting with a number and capital letter)
    next_pattern = r'^\s*(?:\d+\.?\s+)[A-Z][a-zA-Z\s]+'
    matches = list(re.finditer(next_pattern, text, re.MULTILINE))
    end_idx = len(text)
    for m in matches:
        if m.start() > start_idx:
            end_idx = m.start()
            break
    return text[start_idx:end_idx].strip()

for f in files:
    print("="*60)
    print(f"File: {f}")
    print("="*60)
    with open(f, "r", encoding="utf-8") as file_in:
        content = file_in.read()
    
    print("\n--- Abstract / Intro (First 4000 chars) ---")
    print(content[:4000])
    
    print("\n--- Proposed Method / Methodology Section ---")
    method_sec = extract_section(content, "Method")
    if method_sec == "Section not found.":
        method_sec = extract_section(content, "Proposed Method")
    if method_sec == "Section not found.":
        method_sec = extract_section(content, "Methodology")
    print(method_sec[:3000])
    
    print("\n--- Experiments Section ---")
    exp_sec = extract_section(content, "Experiments")
    if exp_sec == "Section not found.":
        exp_sec = extract_section(content, "Experimental")
    print(exp_sec[:3000])
