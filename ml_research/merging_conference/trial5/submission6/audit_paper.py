import re

def audit_tex(file_path):
    print(f"Auditing LaTeX file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    errors_found = 0
    
    # 1. Search for duplicate words (case-insensitive, ignoring LaTeX commands/newlines)
    duplicate_patterns = [
        r'\b(the)\s+(the)\b',
        r'\b(a)\s+(a)\b',
        r'\b(an)\s+(an)\b',
        r'\b(of)\s+(of)\b',
        r'\b(in)\s+(in)\b',
        r'\b(is)\s+(is)\b',
        r'\b(that)\s+(that)\b',
        r'\b(to)\s+(to)\b',
        r'\b(and)\s+(and)\b',
        r'\b(we)\s+(we)\b',
        r'\b(our)\s+(our)\b',
        r'\b(it)\s+(it)\b'
    ]
    
    print("\n[1] Checking for duplicate adjacent words...")
    for pattern in duplicate_patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        for idx, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('%'):
                continue
            matches = list(compiled.finditer(line))
            for match in matches:
                print(f"  Line {idx+1}: Found duplicate '{match.group(0)}' in: \"{line.strip()[:80]}...\"")
                errors_found += 1
                
    # 2. Check for unmatched brackets/parentheses in non-comment lines
    print("\n[2] Checking for unmatched brackets...")
    braces_open = 0
    brackets_open = 0
    parens_open = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith('%'):
            continue
        # Strip escaped characters like \{ or \}
        clean_line = line.replace(r'\{', '').replace(r'\}', '').replace(r'\[', '').replace(r'\]', '')
        # Check braces
        for char in clean_line:
            if char == '{': braces_open += 1
            elif char == '}': braces_open -= 1
            elif char == '[': brackets_open += 1
            elif char == ']': brackets_open -= 1
            elif char == '(': parens_open += 1
            elif char == ')': parens_open -= 1
            
    if braces_open != 0:
        print(f"  Warning: Unbalanced braces (curly {{}}): {braces_open}")
    else:
        print("  Curly braces are balanced.")
        
    # 3. Verify all \label and \ref matching
    print("\n[3] Checking for reference matching...")
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', content))
    crefs = set(re.findall(r'\\cref\{([^}]+)\}', content))
    Crefs = set(re.findall(r'\\Cref\{([^}]+)\}', content))
    
    all_refs = refs.union(crefs).union(Crefs)
    
    missing_labels = all_refs - labels
    if missing_labels:
        for ref in missing_labels:
            print(f"  Warning: Reference '{ref}' has no corresponding label in paper.tex!")
            errors_found += 1
    else:
        print(f"  All {len(all_refs)} references have corresponding labels ({len(labels)} labels found).")
        
    # 4. Search for citation markers and check bib keys if possible
    print("\n[4] Checking for citations...")
    citations = set()
    for match in re.finditer(r'\\cite\{([^}]+)\}', content):
        for cite in match.group(1).split(','):
            citations.add(cite.strip())
            
    # Read bib file
    try:
        with open("example_paper.bib", "r", encoding='utf-8') as f_bib:
            bib_content = f_bib.read()
        bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib_content))
        
        missing_bib_keys = citations - bib_keys
        if missing_bib_keys:
            for key in missing_bib_keys:
                print(f"  Warning: Citation key '{key}' has no entry in example_paper.bib!")
                errors_found += 1
        else:
            print(f"  All {len(citations)} citations have corresponding keys in the bib file.")
    except Exception as e:
         print(f"  Could not read example_paper.bib for validation: {e}")
         
    print(f"\nAudit complete. Errors/Warnings found: {errors_found}")

if __name__ == "__main__":
    audit_tex("paper.tex")
